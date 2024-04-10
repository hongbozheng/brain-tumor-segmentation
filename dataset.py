from torch.utils.data import Dataset
from monai.transforms import apply_transform


class BraTS(Dataset):
    def __init__(self, data, transform=None) -> None:
        self.data = data
        self.transform = transform

        return

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        data = self.data[idx]
        if self.transform:
            data = apply_transform(transform=self.transform, data=data)

        return data