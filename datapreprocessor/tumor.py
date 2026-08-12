import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


class TumorDataset(Dataset):
    """
    Dataset supporting both:

        TUMOR4:
            (3, 32, 32)

        TUMOR4:
            (3, 128, 128)

    Pickle format:

        {
            'data': [...],
            'labels': [...]
        }

    Internally, images are stored as:

        (N, H, W, 3)
    """

    def __init__(
        self,
        root,
        train=True,
        download=False,
        transform=None
    ):

        self.root = root
        self.train = train
        self.transform = transform

        # -------------------------------------------------
        # File names
        # -------------------------------------------------

        if train:
            filenames = [
                "tumor4train.pkl"
                #"tumor4_train.pkl"
            ]
        else:
            filenames = [
                "tumor4test.pkl"
                #"tumor4_test.pkl"
            ]

        # -------------------------------------------------
        # Load data
        # -------------------------------------------------

        all_data = []
        all_labels = []

        for filename in filenames:

            file_path = os.path.join(root, filename)

            if not os.path.exists(file_path):
                raise FileNotFoundError(
                    f"Dataset not found: {file_path}"
                )

            with open(file_path, "rb") as f:
                dataset = pickle.load(f)

            # -------------------------------------------------
            # Validate pickle structure
            # -------------------------------------------------

            if not isinstance(dataset, dict):
                raise ValueError(
                    f"{filename} must contain a dictionary."
                )

            if "data" not in dataset:
                raise ValueError(
                    f"{filename} does not contain 'data'."
                )

            if "labels" not in dataset:
                raise ValueError(
                    f"{filename} does not contain 'labels'."
                )

            data = np.asarray(dataset["data"])

            labels = np.asarray(dataset["labels"])

            # -------------------------------------------------
            # Validate number of samples
            # -------------------------------------------------

            if len(data) != len(labels):
                raise ValueError(
                    f"{filename}: number of data samples "
                    f"({len(data)}) does not match number "
                    f"of labels ({len(labels)})."
                )

            # -------------------------------------------------
            # Validate image dimensions
            #
            # Supports:
            #
            #     N x H x W x 3
            #
            # where H and W can be 32 or 128.
            # -------------------------------------------------

            if data.ndim != 4:
                raise ValueError(
                    f"{filename}: expected data with 4 "
                    f"dimensions, but got {data.shape}."
                )

            # -------------------------------------------------
            # Handle CHW format:
            #
            #     N x 3 x H x W
            #
            # Convert to:
            #
            #     N x H x W x 3
            # -------------------------------------------------

            if data.shape[1] == 3:

                if data.shape[2] != data.shape[3]:
                    raise ValueError(
                        f"{filename}: expected square images, "
                        f"but got {data.shape}."
                    )

                data = np.transpose(
                    data,
                    (0, 2, 3, 1)
                )

            # -------------------------------------------------
            # Now data should be:
            #
            #     N x H x W x 3
            #
            # -------------------------------------------------

            if data.shape[3] != 3:
                raise ValueError(
                    f"{filename}: expected 3 channels, "
                    f"but got shape {data.shape}."
                )

            if data.shape[1] != data.shape[2]:
                raise ValueError(
                    f"{filename}: expected square images, "
                    f"but got {data.shape[1:3]}."
                )

            if data.shape[1] not in [32, 128]:
                raise ValueError(
                    f"{filename}: supported image sizes are "
                    f"32x32 or 128x128, but got "
                    f"{data.shape[1]}x{data.shape[2]}."
                )

            all_data.append(data)
            all_labels.append(labels)

        # -------------------------------------------------
        # Combine data
        # -------------------------------------------------

        self.data = np.concatenate(all_data,axis=0)

        self.targets = np.concatenate(all_labels,axis=0)

        # -------------------------------------------------
        # Convert labels to tensor
        # -------------------------------------------------

        self.targets = torch.tensor(self.targets, dtype=torch.long)

        # -------------------------------------------------
        # Normalize label numbering
        # -------------------------------------------------

        unique_labels = sorted(
            torch.unique(
                self.targets
            ).tolist()
        )

        print(
            f"Original labels: {unique_labels}"
        )

        if unique_labels != list(
            range(len(unique_labels))
        ):

            label_mapping = {
                old_label: new_label
                for new_label, old_label
                in enumerate(unique_labels)
            }

            self.targets = torch.tensor(
                [
                    label_mapping[int(label)]
                    for label in self.targets.tolist()
                ],
                dtype=torch.long
            )

            print(
                f"Label mapping: {label_mapping}"
            )

        # -------------------------------------------------
        # Dataset information
        # -------------------------------------------------

        self.classes = list(
            range(
                len(
                    torch.unique(
                        self.targets
                    )
                )
            )
        )

        self.num_classes = len(
            self.classes
        )

        self.num_channels = 3

        # Detect image size automatically
        self.num_dims = self.data.shape[1]

        print(
            f"TumorDataset: "
            f"{len(self.data)} samples, "
            f"{self.num_classes} classes, "
            f"{self.num_channels} channels, "
            f"{self.num_dims}x{self.num_dims} images"
        )

        print(
            f"Data shape: {self.data.shape}"
        )

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):

        # -------------------------------------------------
        # Get sample and label
        #
        # x:
        #     (32, 32, 3)
        #
        # or:
        #     (128, 128, 3)
        # -------------------------------------------------

        x = self.data[index]
        y = self.targets[index]

        # -------------------------------------------------
        # Ensure uint8
        # -------------------------------------------------

        if x.dtype != np.uint8:

            x = np.clip(
                x,
                0,
                255
            ).astype(
                np.uint8
            )

        # -------------------------------------------------
        # Apply transform
        #
        # ToTensor converts:
        #
        #     (H, W, 3)
        #
        # to:
        #
        #     (3, H, W)
        #
        # Therefore:
        #
        #     32x32 -> 3x32x32
        #
        #     128x128 -> 3x128x128
        # -------------------------------------------------

        if self.transform:

            x = self.transform(x)

        else:

            x = torch.from_numpy(
                x
            ).permute(
                2,
                0,
                1
            ).float() / 255.0

        return x, y