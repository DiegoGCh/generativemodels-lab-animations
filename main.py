"""Lab 3 — Main entry point for training and animation generation."""
import argparse
import os
import sys
import torch


def get_distribution(name: str):
    from distributions import DISTRIBUTIONS
    if name not in DISTRIBUTIONS:
        sys.exit(f"Error: Unknown distribution '{name}'. Valid: {list(DISTRIBUTIONS.keys())}")
    return DISTRIBUTIONS[name]()


def get_process(name: str):
    from processes import PROCESSES
    if name not in PROCESSES:
        sys.exit(f"Error: Unknown process '{name}'. Valid: {list(PROCESSES.keys())}")
    return PROCESSES[name]()


def cmd_train(args):
    import torch
    from models import ScoreNet, VelocityNet
    from training import DiffusionTrainer, FlowMatchingTrainer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dist = get_distribution(args.distribution)
    torch.manual_seed(args.seed)

    if args.model == "diffusion":
        process = get_process(args.process)
        model = ScoreNet(process, parametrization=args.parametrization)
        trainer = DiffusionTrainer(
            model, process, dist,
            n_epochs=args.n_epochs, batch_size=args.batch_size,
            lr=args.lr, t_eps=1e-5, seed=args.seed, device=device
        )
        trainer.train(log_every=max(1, args.n_epochs // 20))
        ckpt = os.path.join("checkpoints",
                            f"{args.distribution}_diffusion_{args.process}_{args.parametrization}")
        trainer.save(ckpt)

    elif args.model == "flow_matching":
        model = VelocityNet()
        trainer = FlowMatchingTrainer(
            model, dist,
            n_epochs=args.n_epochs, batch_size=args.batch_size,
            lr=args.lr, seed=args.seed, device=device
        )
        trainer.train(log_every=max(1, args.n_epochs // 20))
        ckpt = os.path.join("checkpoints", f"{args.distribution}_flow_matching")
        trainer.save(ckpt)
    else:
        sys.exit(f"Error: Unknown model '{args.model}'. Valid: diffusion, flow_matching")


def cmd_animate(args):
    import torch
    import matplotlib
    matplotlib.use("Agg")
    from models import ScoreNet, VelocityNet
    from training import DiffusionTrainer, FlowMatchingTrainer
    from animations import (
        animate_forward_comparison, animate_density_evolution,
        animate_forward_trajectories, animate_score_field,
        animate_reverse_sde, animate_probability_flow,
        animate_flow_matching, animate_step_comparison,
    )
    from distributions import DISTRIBUTIONS
    from processes import PROCESSES

    if args.animation is None:
        sys.exit("Error: --animation N (1-8) is required for animate mode")
    if not 1 <= args.animation <= 8:
        sys.exit(f"Error: --animation must be 1-8, got {args.animation}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("outputs", exist_ok=True)
    dist = get_distribution(args.distribution)

    anim = args.animation

    # Animations 4-8 need a trained model
    if anim in (4, 5, 6, 7, 8):
        if args.checkpoint is None:
            sys.exit("Error: --checkpoint is required for animations 4-8")
        if not os.path.exists(args.checkpoint):
            sys.exit(f"Error: checkpoint not found: {args.checkpoint}")

    if anim in (1, 2, 3, 4, 5, 6) or (anim == 8 and args.model == "diffusion"):
        process = get_process(args.process)

    if anim == 1:
        from processes import VPProcess, VEProcess, SubVPProcess
        animate_forward_comparison(
            dist, [VPProcess(), VEProcess(), SubVPProcess()], ["VP", "VE", "Sub-VP"],
            n_samples=400, n_frames=50, fps=20,
            out_path=f"outputs/1_forward_comparison_{args.distribution}.mp4"
        )

    elif anim == 2:
        dists = [DISTRIBUTIONS[n]() for n in list(DISTRIBUTIONS.keys())[:2]]
        animate_density_evolution(
            dists, [process], [args.process],
            n_samples=800, n_frames=40, fps=15,
            out_path=f"outputs/2_density_evolution_{args.distribution}.mp4"
        )

    elif anim == 3:
        animate_forward_trajectories(
            dist, process, n_particles=50, n_frames=60, fps=20,
            out_path=f"outputs/3_forward_trajectories_{args.distribution}_{args.process}.mp4"
        )

    elif anim in (4, 5, 6):
        model = ScoreNet(process, parametrization=args.parametrization)
        DiffusionTrainer.load_model(model, args.checkpoint, device=device)
        model = model.to(device).eval()

        if anim == 4:
            animate_score_field(
                dist, process, model, n_samples=200, grid_size=15, n_frames=40, fps=15,
                out_path=f"outputs/4_score_field_{args.distribution}_{args.process}_{args.parametrization}.mp4",
                device=device
            )
        elif anim == 5:
            animate_reverse_sde(
                process, model, n_particles=800, n_steps=args.n_steps, seed=args.seed, fps=20,
                out_path=f"outputs/5_reverse_sde_{args.distribution}_{args.process}_{args.parametrization}_N{args.n_steps}.mp4",
                device=device
            )
        elif anim == 6:
            animate_probability_flow(
                process, model, n_particles=800, n_steps=args.n_steps, seed=args.seed, fps=20,
                sampler_name=args.sampler,
                out_path=f"outputs/6_probability_flow_{args.distribution}_{args.process}_{args.parametrization}_N{args.n_steps}.mp4",
                device=device
            )

    elif anim == 7:
        fm_model = VelocityNet()
        FlowMatchingTrainer.load_model(fm_model, args.checkpoint, device=device)
        fm_model = fm_model.to(device).eval()
        animate_flow_matching(
            fm_model, n_particles=800, n_steps=args.n_steps, seed=args.seed, fps=20, grid_size=15,
            out_path=f"outputs/7_flow_matching_{args.distribution}_N{args.n_steps}.mp4",
            device=device
        )

    elif anim == 8:
        if args.model == "diffusion":
            model = ScoreNet(process, parametrization=args.parametrization)
            DiffusionTrainer.load_model(model, args.checkpoint, device=device)
            model = model.to(device).eval()
            animate_step_comparison(
                process, model, mode="diffusion", n_particles=500, seed=args.seed, fps=15,
                out_path=f"outputs/8_step_comparison_{args.distribution}_diffusion.mp4",
                device=device
            )
        elif args.model == "flow_matching":
            fm_model = VelocityNet()
            FlowMatchingTrainer.load_model(fm_model, args.checkpoint, device=device)
            fm_model = fm_model.to(device).eval()
            animate_step_comparison(
                None, fm_model, mode="flow_matching", n_particles=500, seed=args.seed, fps=15,
                out_path=f"outputs/8_step_comparison_{args.distribution}_flow_matching.mp4",
                device=device
            )


def main():
    parser = argparse.ArgumentParser(
        description="Lab 3 — Generative Models Visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Shared arguments
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--distribution", required=True,
                        choices=["two_moons", "eight_gaussians", "checkerboard",
                                 "spirals", "concentric_rings", "pinwheel"])
    shared.add_argument("--model", required=True, choices=["diffusion", "flow_matching"])
    shared.add_argument("--process", default="vp", choices=["vp", "ve", "sub_vp"])
    shared.add_argument("--parametrization", default="epsilon", choices=["epsilon", "v"])
    shared.add_argument("--seed", type=int, default=42)

    # Train subcommand
    train_p = subparsers.add_parser("train", parents=[shared], help="Train a model")
    train_p.add_argument("--n_epochs", type=int, default=10000)
    train_p.add_argument("--batch_size", type=int, default=4096)
    train_p.add_argument("--lr", type=float, default=3e-4)

    # Animate subcommand
    anim_p = subparsers.add_parser("animate", parents=[shared], help="Generate animation")
    anim_p.add_argument("--animation", type=int, choices=range(1, 9), metavar="N")
    anim_p.add_argument("--sampler", default="euler", choices=["euler", "heun", "em"])
    anim_p.add_argument("--n_steps", type=int, default=100)
    anim_p.add_argument("--checkpoint", default=None)
    anim_p.add_argument("--preview", action="store_true",
                        help="Save 5 key-frame PNGs to preview/ instead of rendering .mp4")

    args = parser.parse_args()

    if args.mode == "train":
        cmd_train(args)
    elif args.mode == "animate":
        if getattr(args, "preview", False):
            from utils.preview import run_preview
            run_preview(lambda: cmd_animate(args))
        else:
            cmd_animate(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
