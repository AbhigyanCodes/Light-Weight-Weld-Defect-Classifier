# Dataset layout and usage

Place images and a labels CSV in `data/` as follows:
```bash
data/
├── images/
│   ├── img_0001.jpg
│   └── ...
└── labels.csv
```

labels.csv format:
filename,label
data/images/img_0001.jpg,1
data/images/img_0002.jpg,0

- label: 1 => welded, 0 => non-welded
- The scripts expect `filename` to be a path relative to repository root or an absolute path.

Small sample images (safe to check-in) can go to `data/samples/`.
Large raw datasets should be kept out of git (use `datasets/get_data.sh`).
