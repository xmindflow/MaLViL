# Pretrained encoder weights

MaLViL uses a **PVTv2-B2** encoder initialized from ImageNet.

Download `pvt_v2_b2.pth` from the official PVT release
([v2 tag](https://github.com/whai362/PVT/releases/tag/v2), Classification model zoo)
and place it here:

```
pretrained_pth/pvt/pvt_v2_b2.pth
```

```bash
mkdir -p pretrained_pth/pvt
wget -O pretrained_pth/pvt/pvt_v2_b2.pth \
  https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth
```

The launcher passes `--encoder_ptdir pretrained_pth`. The ImageNet file is not
redistributed (third-party license). Dataset-specific MaLViL heads live on the
[GitHub release](https://github.com/xmindflow/MaLViL/releases), not in this folder.
