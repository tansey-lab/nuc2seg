import logging
import os
from os.path import join
from pathlib import Path

import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def xenium_collate_fn(data):
    outputs = {key: [] for key in data[0].keys()}
    for sample in data:
        for key, val in sample.items():
            outputs[key].append(val)
    outputs['X'] = pad_sequence(outputs['X'], batch_first=True, padding_value=-1)
    outputs['Y'] = pad_sequence(outputs['Y'], batch_first=True, padding_value=-1)
    outputs['gene'] = pad_sequence(outputs['gene'], batch_first=True, padding_value=-1)
    outputs['labels'] = torch.stack(outputs['labels'])
    outputs['angles'] = torch.stack(outputs['angles'])
    outputs['classes'] = torch.stack(outputs['classes'])
    outputs['label_mask'] = torch.stack(outputs['label_mask']).type(torch.bool)
    outputs['nucleus_mask'] = torch.stack(outputs['nucleus_mask']).type(torch.bool)
    outputs['location'] = torch.stack(outputs['location']).type(torch.long)

    # Edge case: pad_sequence will squeeze tensors if there are no entries.
    # In that case, we just need to add the dimension back.
    if len(outputs['gene'].shape) == 1:
        outputs['X'] = outputs['X'][:,None]
        outputs['Y'] = outputs['Y'][:,None]
        outputs['gene'] = outputs['gene'][:,None]

    return outputs

class XeniumDataset(Dataset):
    def __init__(self, tiles_dir: str):
        self.transcripts_dir = Path(join(tiles_dir, 'transcripts/'))
        self.labels_dir = Path(join(tiles_dir, 'labels/'))
        self.angles_dir = Path(join(tiles_dir, 'angles/'))
        self.classes_dir = Path(join(tiles_dir, 'classes/'))

        self.locations = np.load(join(tiles_dir, 'locations.npy'))

        self.ids = np.arange(self.locations.shape[0])

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        self.class_counts = np.load(join(tiles_dir, 'class_counts.npy'))
        self.transcript_counts = np.load(join(tiles_dir, 'transcript_counts.npy'))
        self.max_length = self.transcript_counts.max()
        self.label_values = np.arange(self.class_counts.shape[1])-1
        self.n_classes = self.class_counts.shape[1]-2
        self.gene_ids = {int(i): j for i,j in np.load(join(tiles_dir, 'gene_ids.npy'))}
        self.n_genes = max(self.gene_ids)+1

        # Note: class IDs are 1-based since ID=0 is background
        logging.info(f'Unique label values: {self.label_values}')


    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        transcripts_file = os.path.join(self.transcripts_dir, f'{idx}.npz')
        labels_file = os.path.join(self.labels_dir, f'{idx}.npz')
        angles_file = os.path.join(self.angles_dir, f'{idx}.npz')
        classes_file = os.path.join(self.classes_dir, f'{idx}.npz')


        xyg = np.load(transcripts_file)['arr_0']
        labels = np.load(labels_file)['arr_0']
        angles = np.load(angles_file)['arr_0']
        classes = np.load(classes_file)['arr_0']
        labels_mask = labels > -1
        nucleus_mask = labels > 0

        return {
                'X': torch.as_tensor(np.array(xyg[:,0])).long().contiguous(),
                'Y': torch.as_tensor(np.array(xyg[:,1])).long().contiguous(),
                'gene': torch.as_tensor(np.array(xyg[:,2])).long().contiguous(),
                'labels': torch.as_tensor(labels).long().contiguous(),
                'angles': torch.as_tensor(angles).float().contiguous(),
                'classes': torch.as_tensor(classes).long().contiguous(),
                'label_mask': torch.as_tensor(labels_mask).bool().contiguous(),
                'nucleus_mask': torch.as_tensor(nucleus_mask).bool().contiguous(),
                'location': self.locations[idx]
        }
