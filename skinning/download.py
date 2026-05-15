import os

from huggingface_hub import hf_hub_download

workdir = os.path.dirname(os.path.abspath(__file__))

link_source = os.path.abspath(os.path.join(workdir, "../skeleton/third_partys/Michelangelo"))
if not os.path.isdir(link_source):
    raise FileNotFoundError(link_source)
link_target = os.path.join(workdir, "third_partys", "Michelangelo")
link_target_dir = os.path.dirname(link_target)
os.makedirs(link_target_dir, exist_ok=True)
if os.path.lexists(link_target):
    os.remove(link_target)
os.symlink(os.path.relpath(link_source, link_target_dir), link_target)

file_path = hf_hub_download(
    repo_id="mikaelaangel/partfield-ckpt",
    filename="model_objaverse.ckpt",
    local_dir=os.path.join(workdir, "third_partys/PartField/ckpt"),
)

file_path = hf_hub_download(
    repo_id="Seed3D/Puppeteer",
    filename="skinning_ckpts/puppeteer_skin_w_diverse_pose_depth1.pth",
    local_dir=workdir,
)

file_path = hf_hub_download(
    repo_id="Seed3D/Puppeteer",
    filename="skinning_ckpts/puppeteer_skin_w_diverse_pose_depth2.pth",
    local_dir=workdir,
)

file_path = hf_hub_download(
    repo_id="Seed3D/Puppeteer",
    filename="skinning_ckpts/puppeteer_skin_wo_diverse_pose_depth1.pth",
    local_dir=workdir,
)
