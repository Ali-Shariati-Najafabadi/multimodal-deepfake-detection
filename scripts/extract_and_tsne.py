import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# t-SNE der Video- und Audio-Embeddings, eingefaerbt nach Datensatz.
# Laeuft auf dem Server und nutzt die vorhandene Modell-/Loader-Infrastruktur.
# pip install scikit-learn
#
# Anpassen: Import und Aufbau des Modells sowie der DataLoader je Datensatz.
# Das Modell muss die Feature-Vektoren vor der Fusion liefern (Video 512, Audio 768).

# from models.phase1_model_updated import Phase1Model
# from data.multi_dataset_loader import build_loader

@torch.no_grad()
def collect(model, loader, device, max_batches=30):
    vids, auds = [], []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        video = batch["video"].to(device)
        audio = batch["audio"].to(device)
        # TODO: an die eigene Modell-API anpassen. Erwartet werden die
        # Feature-Vektoren vor der Fusion, z.B. ueber return_features=True.
        out = model(video, audio, return_features=True)
        vids.append(out["video_features"].cpu().numpy())
        auds.append(out["audio_features"].cpu().numpy())
    return np.concatenate(vids), np.concatenate(auds)

def tsne_plot(ax, feats, labels, names, title):
    emb = TSNE(n_components=2, init="pca", perplexity=30, random_state=42).fit_transform(feats)
    for k, name in enumerate(names):
        m = labels == k
        ax.scatter(emb[m, 0], emb[m, 1], s=6, alpha=0.6, label=name)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, markerscale=2)
    ax.set_xticks([]); ax.set_yticks([])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="tsne_embeddings")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # TODO: Modell laden und je Datensatz einen Loader bauen.
    # model = Phase1Model(num_classes=4).to(device).eval()
    # model.load_state_dict(torch.load(args.checkpoint, map_location=device)["model_state_dict"])
    # loaders = {"AVD": build_loader("av_deepfake"), "FakeAVCeleb": build_loader("fakeavceleb"),
    #            "DeepSpeak": build_loader("deepspeak")}
    raise SystemExit("Bitte Modell-Import und Loader oben eintragen (TODO-Markierungen).")

    names = list(loaders)
    V, A, L = [], [], []
    for k, name in enumerate(names):
        v, a = collect(model, loaders[name], device)
        V.append(v); A.append(a); L.append(np.full(len(v), k))
    V, A, L = np.concatenate(V), np.concatenate(A), np.concatenate(L)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    tsne_plot(axes[0], V, L, names, "Video-Embeddings")
    tsne_plot(axes[1], A, L, names, "Audio-Embeddings")
    fig.savefig(f"{args.out}.pdf", bbox_inches="tight")
    fig.savefig(f"{args.out}.png", dpi=200, bbox_inches="tight")
    print(f"gespeichert: {args.out}.pdf/.png")

if __name__ == "__main__":
    main()