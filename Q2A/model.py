
import torch, os, json
from torch import nn
from torch.functional import F
from pytorch_lightning import LightningModule
from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR

class MLP(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, in_features)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class Q2A(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.mlp_v = MLP(cfg.INPUT.DIM, cfg.INPUT.DIM)
        self.mlp_t = MLP(cfg.INPUT.DIM, cfg.INPUT.DIM)
        self.mlp_pre = MLP(cfg.INPUT.DIM*(3+cfg.INPUT.NUM_MASKS), cfg.MODEL.DIM_STATE)

        self.s2v = nn.MultiheadAttention(cfg.INPUT.DIM, cfg.MODEL.NUM_HEADS)
        self.qa2s = nn.MultiheadAttention(cfg.INPUT.DIM, cfg.MODEL.NUM_HEADS)
        
        self.state = torch.randn(cfg.MODEL.DIM_STATE, device="cpu")
        if cfg.MODEL.HISTORY.ARCH == "mlp":
            self.proj = MLP(cfg.MODEL.DIM_STATE*2, 1)
        elif cfg.MODEL.HISTORY.ARCH == "gru":
            self.gru = nn.GRUCell(cfg.MODEL.DIM_STATE, cfg.MODEL.DIM_STATE)
            self.proj = MLP(cfg.MODEL.DIM_STATE, 1)
        else:
            assert False, "unknown arch"
        
        self.history_train = cfg.MODEL.HISTORY.TRAIN
        self.history_val = cfg.MODEL.HISTORY.VAL
        self.cfg = cfg

    def forward(self, batch):
        loss, count = 0, 0
        results = []
        for video, script, question, actions, label, meta in batch:
            video = self.mlp_v(video) 
            script = self.mlp_t(script)
            video = self.s2v(script.unsqueeze(1), video.unsqueeze(1), video.unsqueeze(1))[0].squeeze_()
            question = self.mlp_t(question)
            
            state = self.state
            scores = []
            for i, actions_per_step in enumerate(actions):
                a_texts, a_buttons = zip(*[(action['text'], action['button']) for action in actions_per_step])
                a_texts = self.mlp_t(torch.cat(a_texts))
                A = len(a_buttons)
                a_buttons = self.mlp_v(
                    torch.stack(a_buttons).view(A, -1, a_texts.shape[1])
                ).view(A, -1) 
                qa = question + a_texts
                qa_script, qa_script_mask = self.qa2s(
                    qa.unsqueeze(1), script.unsqueeze(1), script.unsqueeze(1)
                )
                qa_video = qa_script_mask @ video
                inputs = torch.cat(
                    [qa_video.view(A, -1), qa_script.view(A, -1), qa.view(A, -1), a_buttons.view(A, -1)],
                    dim=1
                )
                inputs = self.mlp_pre(inputs)
                if hasattr(self, "gru"):
                    states = self.gru(inputs, state.expand_as(inputs))
                else:
                    states = torch.cat([inputs, state.expand_as(inputs)], dim=1)
                logits = self.proj(states)
                if self.training:
                    loss += F.cross_entropy(logits.view(1, -1), label[i].view(-1))
                    count += 1
                else:
                    scores.append(logits.view(-1).tolist())
                if self.history_train == "gt" and self.training:
                    state = inputs[label[i]]
                if (self.history_train == "max" and self.training) \
                    or (self.history_val == "max" and not self.training):
                    state = inputs[logits.argmax()]
            if not self.training:
                meta["scores"] = scores
                results.append(meta)
        if self.training:
            return loss / count
        else:
            return results

models = {"q2a": Q2A}

class ModelModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.model = models[cfg.MODEL.ARCH](cfg)
        self.cfg = cfg
        self.save_hyperparameters()
    
    def training_step(self, batch, idx):
        loss = self.model(batch)
        dataset = self.trainer.datamodule.__class__.__name__
        self.log(f"{dataset} loss", loss, rank_zero_only=True)
        return loss
    
    def configure_optimizers(self):
        cfg = self.cfg
        model = self.model
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.SOLVER.LR,
            momentum=0.9, weight_decay=0.0005) 
        lr_scheduler = LinearWarmupCosineAnnealingLR(optimizer, 
            warmup_epochs=cfg.SOLVER.WARMUP_EPOCHS, max_epochs=cfg.SOLVER.MAX_EPOCHS, 
            warmup_start_lr=cfg.SOLVER.LR*0.1)
        return [optimizer], [lr_scheduler]
    
    def validation_step(self, batch, idx):
        batched_results = self.model(batch)
        return batched_results
    
    # def validation_epoch_end(self, outputs) -> None:
    #     scores, labels, metas = list(zip(*outputs))
    #     scores = sum(scores, [])
    #     labels = sum(labels, [])
    #     metas = sum(metas, [])
    #     if len(labels) > 0: # evaluation
    #         recall_1, recall_3, mean_rank, mrr = [], [], [], []
    #         for score, label in zip(scores, labels):
    #             sorted_indices = score.sort(descending=True)[1]
    #             mask = sorted_indices == label
    #             recall_1.append(mask[0].float())
    #             recall_3.append(mask[:3].float().sum())
    #             mean_rank.append(mask.nonzero().squeeze_().float() + 1)
    #             mrr.append(len(mask) / (mean_rank[-1]))
    #         recall_1 = torch.stack(recall_1).mean()
    #         recall_3 = torch.stack(recall_3).mean()
    #         mean_rank = torch.stack(mean_rank).mean()
    #         mrr = torch.stack(mrr).mean()
    #         dataset = self.trainer.datamodule.__class__.__name__
    #         self.log(f"{dataset} recall@1", recall_1, rank_zero_only=True)
    #         self.log(f"{dataset} recall@3", recall_3, rank_zero_only=True)
    #         self.log(f"{dataset} mean_rank", mean_rank, rank_zero_only=True)
    #         self.log(f"{dataset} mrr", mrr)

    #         print(f"{dataset} recall@1", recall_1)
    #         print(f"{dataset} recall@3", recall_3)
    #         print(f"{dataset} mean_rank", mean_rank)
    #         print(f"{dataset} mrr", mrr)
        
    def validation_epoch_end(self, outputs) -> None:
        from eval_for_loveu_cvpr2022 import evaluate
        results = sum(outputs, [])
        all_preds = {}
        for result in results:
            pred = dict(
                question=result['question'], 
                scores=result['scores']
            )
            folder = result['folder']
            if folder not in all_preds:
                all_preds[folder] = []
            all_preds[folder].append(pred)

        if self.cfg.DATASET.GT:
            with open(self.cfg.DATASET.GT) as f:
                all_annos = json.load(f)
            r1, r3, mr, mrr = evaluate(all_preds, all_annos)
            dataset = self.trainer.datamodule.__class__.__name__
            # for tensorboard
            self.log(f"{dataset} recall@1", r1, rank_zero_only=True)
            self.log(f"{dataset} recall@3", r3, rank_zero_only=True)
            self.log(f"{dataset} mean_rank", mr, rank_zero_only=True)
            self.log(f"{dataset} mrr", mrr)
            # for terminal
            print(f"{dataset} recall@1", r1)
            print(f"{dataset} recall@3", r3)
            print(f"{dataset} mean_rank", mr)
            print(f"{dataset} mrr", mrr) 
        else:
            json_name = f"submit_test_{self.current_epoch}e.json"
            json_file = os.path.join(self.logger.log_dir, json_name)
            if not os.path.exists(self.logger.log_dir):
                os.makedirs(self.logger.log_dir)
            print("\n No ground-truth labels for validation \n")
            print(f"Generating json file at {json_file}. You can zip and submit it to CodaLab ;)")
            with open(json_file, 'w') as f:
                json.dump(all_preds, f)

def build_model(cfg):
    return ModelModule(cfg)
