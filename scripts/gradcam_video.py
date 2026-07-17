import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Grad-CAM fuer den ResNet-50-Video-Zweig (Phase 1).
# Ueberlagert die Aktivierungskarte der letzten Faltungsschicht auf die Frames.
# Laeuft auf dem Server mit dem eigenen Modell.
#
# Anpassen: Modell-Import, Laden eines vorverarbeiteten Video-Tensors (1,3,16,224,224)
# und die Referenz auf die letzte Conv-Schicht (ResNet layer4).

# from models.phase1_model_updated import Phase1Model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="gradcam")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # TODO: Modell laden und video-Tensor bereitstellen.
    # model = Phase1Model(num_classes=2).to(device).eval()
    # model.load_state_dict(torch.load(args.checkpoint, map_location=device)["model_state_dict"])
    # video = load_preprocessed_video(...).to(device)  # (1,3,16,224,224)
    # target_layer = model.video_model.backbone[-1]     # letzte ResNet-Stufe (layer4)
    raise SystemExit("Bitte Modell-Import, Video-Tensor und target_layer eintragen (TODO).")

    acts, grads = {}, {}
    target_layer.register_forward_hook(lambda m, i, o: acts.__setitem__("v", o.detach()))
    target_layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("v", go[0].detach()))

    logits = model.video_forward(video)          # an eigene API anpassen
    score = logits[:, logits.argmax(dim=1)].sum()
    model.zero_grad(); score.backward()

    a, g = acts["v"], grads["v"]                  # (N,C,H,W) je Frame
    weights = g.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * a).sum(dim=1))
    cam = (cam - cam.amin(dim=(1, 2), keepdim=True)) / (cam.amax(dim=(1, 2), keepdim=True) + 1e-6)
    cam = cam.cpu().numpy()

    frames = video[0].permute(1, 2, 3, 0).cpu().numpy()  # (16,224,224,3)
    frames = (frames - frames.min()) / (frames.ptp() + 1e-6)
    idxs = np.linspace(0, len(frames) - 1, 6, dtype=int)

    fig, axes = plt.subplots(1, len(idxs), figsize=(12, 2.4))
    for ax, i in zip(axes, idxs):
        ax.imshow(frames[i])
        heat = np.array(plt.cm.jet(cam[i]))[..., :3]
        ax.imshow(heat, alpha=0.45, extent=(0, 224, 224, 0))
        ax.axis("off")
    fig.savefig(f"{args.out}.pdf", bbox_inches="tight")
    fig.savefig(f"{args.out}.png", dpi=200, bbox_inches="tight")
    print(f"gespeichert: {args.out}.pdf/.png")

if __name__ == "__main__":
    main()