import os
import torch
import torch.optim as optim
import yaml


class DiffusionTrainer:
    def __init__(self, model, process, distribution,
                 n_epochs: int = 10000, batch_size: int = 4096,
                 lr: float = 3e-4, t_eps: float = 1e-5, seed: int = 42,
                 device: str = "cpu"):
        self.model = model.to(device)
        self.process = process
        self.distribution = distribution
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.t_eps = t_eps
        self.device = device
        torch.manual_seed(seed)
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.config = dict(
            n_epochs=n_epochs, batch_size=batch_size, lr=lr,
            t_eps=t_eps, seed=seed,
            distribution=distribution.name,
            process=type(process).__name__,
            parametrization=model.parametrization,
        )

    def train_step(self) -> float:
        self.model.train()
        x0 = self.distribution.sample(self.batch_size).to(self.device)
        t = torch.rand(self.batch_size, device=self.device) * (self.process.T - self.t_eps) + self.t_eps
        xt, epsilon = self.process.sample_forward(x0, t)

        pred = self.model(xt, t)

        if self.model.parametrization == "epsilon":
            loss = (pred - epsilon).pow(2).mean()
        elif self.model.parametrization == "v":
            a_t = self.process.m(t).unsqueeze(-1)
            b_t = self.process.sigma(t).unsqueeze(-1)
            v_target = a_t * epsilon - b_t * x0
            loss = (pred - v_target).pow(2).mean()
        else:
            raise ValueError(f"Unknown parametrization: {self.model.parametrization}")

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
