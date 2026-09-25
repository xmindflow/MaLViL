"""Dataloaders for MaLViL paper datasets: skin lesion, BUSI, Synapse."""

import os
from pathlib import Path

from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _synapse_list_dir(data_dir):
    """Prefer lists next to the data; fall back to repo `splits/synapse/`."""
    if os.path.isfile(os.path.join(data_dir, "train.txt")):
        return data_dir
    repo_splits = _REPO_ROOT / "splits" / "synapse"
    if os.path.isfile(repo_splits / "train.txt"):
        print(f"Using Synapse split lists from {repo_splits}")
        return str(repo_splits)
    raise FileNotFoundError(
        f"No train.txt in {data_dir} or {repo_splits}. "
        "Copy splits/synapse/*.txt into $DATA_DIR/Synapse/."
    )


def get_dataloaders(args, logging):
    tr_loader, vl_loader, te_loader = None, None, None
    name = args.dataset_name.lower()

    if name in ["isic2017", "isic2018", "ph2", "ham", "ham10000"]:
        if "ph2" in args.data_dir.lower():
            from datasets.skin import get_ph2 as get_skin_db
        elif "isic2017" in args.data_dir.lower():
            from datasets.skin import get_isic2017 as get_skin_db
        elif "isic2018" in args.data_dir.lower():
            from datasets.skin import get_isic2018 as get_skin_db
        elif "ham" in args.data_dir.lower():
            from datasets.skin import get_ham10000 as get_skin_db
        else:
            raise ValueError(f"Cannot infer skin dataset from data_dir={args.data_dir}")

        dbs = get_skin_db(args, verbose=True)
        db_train, db_val, db_test = dbs["tr_dataset"], dbs["vl_dataset"], dbs["te_dataset"]
        logging.info(
            f"Training dataset size: {len(db_train)}, val: {len(db_val)}, test: {len(db_test)}"
        )
        print(
            f"Training dataset size: {len(db_train)}, val: {len(db_val)}, test: {len(db_test)}"
        )
        tr_loader = DataLoader(
            db_train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        vl_loader = DataLoader(
            db_val, batch_size=1, shuffle=False, num_workers=1, pin_memory=True
        )
        te_loader = DataLoader(
            db_test, batch_size=1, shuffle=False, num_workers=1, pin_memory=True
        )

    elif name == "busi":
        from datasets.dataset_busi import get_busi

        dbs = get_busi(args, verbose=True)
        db_train, db_val, db_test = (
            dbs["tr_dataset"],
            dbs["vl_dataset"],
            dbs["te_dataset"],
        )
        logging.info(
            f"Training dataset size: {len(db_train)}, val: {len(db_val)}, test: {len(db_test)}"
        )
        print(
            f"Training dataset size: {len(db_train)}, val: {len(db_val)}, test: {len(db_test)}"
        )
        tr_loader = DataLoader(
            db_train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        vl_loader = DataLoader(
            db_val, batch_size=1, shuffle=False, num_workers=1, pin_memory=True
        )
        te_loader = DataLoader(
            db_test, batch_size=1, shuffle=False, num_workers=1, pin_memory=True
        )

    elif name == "synapse":
        from datasets.dataset_synapse import SynapseDatasetFast

        list_dir = _synapse_list_dir(args.data_dir)
        db_train = SynapseDatasetFast(
            base_dir=f"{args.data_dir}/train_npz",
            list_dir=list_dir,
            split="train",
            img_size=args.img_size,
            norm_x_transform=None,
            norm_y_transform=None,
        )
        db_test = SynapseDatasetFast(
            base_dir=f"{args.data_dir}/test_vol_h5",
            list_dir=list_dir,
            split="test",
            img_size=args.img_size,
            norm_x_transform=None,
            norm_y_transform=None,
        )
        tr_loader = DataLoader(
            db_train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        te_loader = DataLoader(
            db_test, batch_size=1, shuffle=False, num_workers=1, pin_memory=True
        )
        logging.info(
            f"Training dataset size: {len(db_train)} 2D-slices, Test: {len(db_test)} 3D-volumes"
        )
        print(
            f"Training dataset size: {len(db_train)} 2D-slices, Test: {len(db_test)} 3D-volumes"
        )

    else:
        raise ValueError(
            f"Unsupported dataset '{args.dataset_name}'. "
            "Release code supports: ISIC2017, ISIC2018, PH2, HAM10000, BUSI, SYNAPSE."
        )

    return tr_loader, vl_loader, te_loader
