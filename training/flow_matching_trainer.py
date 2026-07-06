import os
import torch
import torch.optim as optim
import yaml


class FlowMatchingTrainer:
    def __init__(self, model, distribution,
                 n_epochs: int = 10000, batch_size: int = 4096,
                 lr: float = 3e-4, seed: int = 42, device: str = "cpu"):
        self.model = model.to(device)
        self.distribution = distribution
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device
        torch.manual_seed(seed)
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.config = dict(
            n_epochs=n_epochs, batch_size=batch_size, lr=lr,
            seed=seed, distribution=distribution.name, model="flow_matching",
        )

    def train_step(self) -> float:
        self.model.train()
        x1 = self.distribution.sample(self.batch_size).to(self.device)  # data
        x0 = torch.randn_like(x1)                                        # noise
        t = torch.rand(self.batch_size, device=self.device)

        # Linear interpolation: x_t = (1-t)*x0 + t*x1
        t_view = t.unsqueeze(-1)
        xt = (1 - t_view) * x0 + t_view * x1

        # Target velocity: u = x1 - x0 (constant along trajectory)
        u = x1 - x0

        pred = self.model(xt, t)
        loss = (pred - u).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, log_every: int = 500):
        losses = []
        for epoch in range(1, self.n_epochs + 1):
            loss = self.train_step()
            losses.append(loss)
            if epoch % log_every == 0:
                print(f"  Epoch {epoch}/{self.n_epochs}  loss={loss:.6f}")
        return losses

    def save(self, checkpoint_dir: str):
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(checkpoint_dir, "model.pt"))
        with open(os.path.join(checkpoint_dir, "config.yaml"), "w") as f:
            yaml.dump(self.config, f)
        print(f"  Checkpoint saved to {checkpoint_dir}")

    @staticmethod
    def load_model(model, checkpoint_dir: str, device: str = "cpu"):
        state = torch.load(os.path.join(checkpoint_dir, "model.pt"), map_location=device)
        model.load_state_dict(state)
        return model
