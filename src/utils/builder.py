"""Build the MaLViL segmentation network (paper release defaults)."""

from networks.malvil.net import DEFAULT_STAGE_CFG, MaLViLNet


def build_model(args, device, writer=None, logger=None):
    if logger is None:
        import logging
        logger = logging.getLogger("malvil")

    stage_cfg = dict(DEFAULT_STAGE_CFG)
    logger.info(f"Using MaLViL stage config: {stage_cfg}")
    print(f"Using MaLViL stage config: {stage_cfg}")

    net = MaLViLNet(
        in_chs=args.input_channels,
        n_classes=args.num_classes,
        chs=[64, 128, 320, 512],
        img_res=[args.img_size, args.img_size],
        drop_prob=0.05,
        encoder="pvt_v2_b2",
        base_ptbbdir=args.encoder_ptdir,
        num_rotations=2,
        share_salvil_weights=True,
        decoder_stage_cfg=stage_cfg,
        logger=logger,
    ).to(device)
    return net
