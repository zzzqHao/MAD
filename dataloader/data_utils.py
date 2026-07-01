import numpy as np
import torch
from dataloader.sampler import CategoriesSampler
import utils
from dataloader.autoaugment import CIFAR10Policy, Cutout, ImageNetPolicy
from torchvision import transforms
from torchvision.transforms import InterpolationMode

DEFAULT_SPLITS = {
    "cifar100": {"base_class": 60, "num_classes": 100, "way": 5, "shot": 5},
    "cub200": {"base_class": 100, "num_classes": 200, "way": 10, "shot": 5},
    "mini_imagenet": {"base_class": 60, "num_classes": 100, "way": 5, "shot": 5},
    "mini_imagenet1s": {"base_class": 60, "num_classes": 100, "way": 5, "shot": 1},
    "ardataset": {"base_class": 16, "num_classes": 32, "way": 4, "shot": 5},
}


def _set_split_args(args):
    defaults = DEFAULT_SPLITS[args.dataset]
    requested_way = getattr(args, "way", None)
    requested_shot = getattr(args, "shot", None)

    args.base_class = defaults["base_class"]
    args.num_classes = defaults["num_classes"]
    args.default_way = defaults["way"]
    args.default_shot = defaults["shot"]
    args.way = requested_way if requested_way is not None else defaults["way"]
    args.shot = requested_shot if requested_shot is not None else defaults["shot"]

    assert args.way > 0, "way must be positive"
    assert args.shot > 0, "shot must be positive"
    assert (args.num_classes - args.base_class) % args.way == 0, (
        "num_classes - base_class must be divisible by way"
    )
    args.sessions = 1 + (args.num_classes - args.base_class) // args.way
    return args


def get_incremental_index(args, session):
    if args.dataset != "cifar100":
        return "data/index_list/" + args.dataset + "/session_" + str(session + 1) + ".txt"

    start_class = (session - 1) * args.way
    end_class = start_class + args.way
    class_groups = []
    source_sessions = 1 + (args.num_classes - args.base_class) // args.default_way
    for source_session in range(1, source_sessions):
        txt_path = "data/index_list/" + args.dataset + "/session_" + str(source_session + 1) + ".txt"
        session_indices = open(txt_path).read().splitlines()
        session_groups = np.array(session_indices).reshape(args.default_way, args.default_shot)
        class_groups.extend([group.tolist() for group in session_groups])

    if end_class > len(class_groups):
        raise ValueError(
            f"Not enough CIFAR100 index entries for session={session}, way={args.way}, shot={args.shot}. "
            f"Need class groups [{start_class}:{end_class}], but only {len(class_groups)} are available."
        )

    selected = []
    for group in class_groups[start_class:end_class]:
        if args.shot > len(group):
            raise ValueError(
                f"shot={args.shot} exceeds available CIFAR100 shots per class ({len(group)}) in index files."
            )
        selected.extend(group[:args.shot])
    return selected


def imagenet_test_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize([image_size, image_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def set_up_datasets(args):
    if args.dataset == 'cifar100':
        import dataloader.cifar100.cifar as Dataset
    if args.dataset == 'cub200':
        import dataloader.cub200.cub200 as Dataset
    if args.dataset == 'mini_imagenet':
        import dataloader.miniimagenet.miniimagenet as Dataset
    if args.dataset == 'mini_imagenet1s':
        import dataloader.miniimagenet.miniimagenet as Dataset
    if args.dataset == 'ardataset':
        import dataloader.ardataset.ardataset as Dataset
    args = _set_split_args(args)
    args.Dataset=Dataset
    if args.exemplars_count == -1:
        args.exemplars_count = args.shot
    assert args.exemplars_count <= args.shot, "Exemplars count cannot be greater than the number of shots in your few shot data"
    return args

def get_dataloader(args,session):
    if session == 0:
        trainset, trainloader, testloader = get_base_dataloader(args)
    else:
        trainset, trainloader, testloader = get_new_dataloader(args)
    return trainset, trainloader, testloader

def get_base_dataloader(args, debug = False, dino_transform = None):
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(0 + 1) + '.txt'
    class_index = np.arange(args.base_class if not debug else 5) # test on a small dataset for debugging purpose
    if args.dataset == 'cifar100':
        trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=True,
                                         index=class_index, base_sess=True)

        testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                        index=class_index, base_sess=True)

    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True,
                                       index=class_index, base_sess=True,  
                                       dino_transform = dino_transform)
        testset = args.Dataset.CUB200(root=args.dataroot, train=False, index=class_index)

    if args.dataset == 'mini_imagenet':
        #默认不做transform
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True,
                                             index=class_index, base_sess=True)
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False, index=class_index)
        trainset.transform = imagenet_test_transform()
        testset.transform = imagenet_test_transform()
    
    if args.dataset == 'ardataset':
        file_list = "data/index_list/" + args.dataset + "/session_" + str(1) + '.txt'
        trainset = args.Dataset.ARdatasets(root=args.dataroot, num_segments=args.num_segments, file_list=file_list, train=True, session=0)
        testset = args.Dataset.ARdatasets(root=args.dataroot, num_segments=args.num_segments, file_list=args.test_file_list, train=False, session=0)


    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_base, shuffle=True,
                                              num_workers=args.num_workers, pin_memory=True)
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.batch_size_test, shuffle=False, 
                                             num_workers=args.num_workers, pin_memory=True)
    return trainset, trainloader, testloader


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

class MultiCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform, n_views = 2):
        self.transform = transform
        self.n_views = n_views

    def __call__(self, x):
        out = []
        for i in range(self.n_views):
            out.append(self.transform(x))
        return out


def get_new_dataloader(args, session, dino_transform = None):
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(session + 1) + '.txt'
    if args.dataset == 'cifar100':
        class_index = get_incremental_index(args, session)
        trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=False,
                                         index=class_index, base_sess=False, way=args.way, shot=args.shot)
    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True,
                                       index_path=txt_path, dino_transform = dino_transform)
    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True,
                                       index_path=txt_path)
        trainset.transform = imagenet_test_transform()
    if args.dataset == 'ardataset':
        file_list = "data/index_list/" + args.dataset + "/session_" + str(session + 1) + '.txt'
        trainset = args.Dataset.ARdatasets(root=args.dataroot, num_segments=args.num_segments, file_list=file_list, train=True, session=session)
    
    # if args.batch_size_new == 0:
    #     batch_size_new = trainset.__len__()
    #     trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_new, shuffle=False,
    #                                               num_workers=args.num_workers, pin_memory=True)
    # else:
    #     trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_new, shuffle=True,
    #                                               num_workers=args.num_workers, pin_memory=True)
    
    batch_size_new = trainset.__len__()
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_new, shuffle=False,
                                                num_workers=args.num_workers, pin_memory=True)

    # test on all encountered classes
    class_new = get_session_classes(args, session)

    if args.dataset == 'cifar100':
        testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                        index=class_new, base_sess=False)
    if args.dataset == 'cub200':
        testset = args.Dataset.CUB200(root=args.dataroot, train=False,
                                      index=class_new)
    if args.dataset == 'mini_imagenet':
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False,
                                      index=class_new)
        testset.transform = imagenet_test_transform()
    if args.dataset == 'ardataset':
        testset = args.Dataset.ARdatasets(root=args.dataroot, num_segments=args.num_segments, file_list=args.test_file_list, train=False, session=session)

    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.batch_size_test, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, testloader

def get_session_classes(args,session):
    class_list=np.arange(args.base_class + session * args.way)
    return class_list

def get_baseline_base_train_dataloader(args, session=0):
    
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(0 + 1) + '.txt'
    class_index = np.arange(args.base_class) # test on a small dataset for debugging purpose
    if args.dataset == 'cifar100':
        normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        if args.rand_aug_sup_con:
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=224, scale=(args.min_crop_scale, 1.)),
                transforms.RandomHorizontalFlip(),
                CIFAR10Policy(),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                Cutout(n_holes=1, length=16),
                normalize,
            ])
        else:
            if args.simple_aug:
                train_transform = transforms.Compose([
                    transforms.RandomResizedCrop(size=224, scale=(args.min_crop_scale, 1.)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
            else:
                train_transform = transforms.Compose([
                    transforms.RandomResizedCrop(size=224, scale=(args.min_crop_scale, 1.)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                    ], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor(),
                    normalize,
                ])
            
            
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=False,
                                         index=class_index, base_sess=True, transform = MultiCropTransform(train_transform))
        testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                        index=class_index, base_sess=False)

    if args.dataset == 'cub200':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if args.rand_aug_sup_con:
            train_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                ImageNetPolicy(),
                transforms.ToTensor(),
                normalize
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(size=224, scale=(args.min_crop_scale, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                normalize,
            ])
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True,
                                       index=class_index, base_sess=True,
                                       dino_transform = None, transform = MultiCropTransform(train_transform))
        testset = args.Dataset.CUB200(root=args.dataroot, train=False, index=class_index)

    if args.dataset == 'mini_imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if args.rand_aug_sup_con:
            train_transform = transforms.Compose([
                transforms.RandAugment(num_ops = 3, magnitude=11),
                transforms.RandomResizedCrop(size=224, scale=(args.min_crop_scale, 1.)),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=224, scale=(args.min_crop_scale, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                normalize,
            ])
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True,
                                             index=class_index, base_sess=True, 
                                             transform = TwoCropTransform(train_transform))
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False, index=class_index)
        testset.transform = imagenet_test_transform()
    
    if args.dataset == 'ardataset':
        file_list = "data/index_list/" + args.dataset + "/session_" + str(session + 1) + '.txt'
        trainset = args.Dataset.ARdatasets(root=args.dataroot, num_segments=args.num_segments, file_list=file_list, train=True, session=session)
        testset = args.Dataset.ARdatasets(root=args.dataroot, num_segments=args.num_segments, file_list=args.test_file_list, train=False, session=session)
    
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_train_base, shuffle=True,
                                            num_workers=args.num_workers, pin_memory=True, drop_last = args.drop_last_batch)
    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.batch_size_test, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    return trainset, trainloader, testloader


def get_baseline_replay_dataloader(args, session):
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(session + 1) + '.txt'
    class_index = np.arange(args.base_class) # test on a small dataset for debugging purpose
    if args.dataset == 'cifar100':        
        normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        if args.rand_aug_sup_con:
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=224, scale=(args.min_crop_scale, 1.)),
                transforms.RandomHorizontalFlip(),
                CIFAR10Policy(),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                Cutout(n_holes=1, length=16),
                normalize,
            ])
        else:
            if args.simple_aug:
                train_transform = transforms.Compose([
                    transforms.RandomResizedCrop(size=224, scale=(args.min_crop_scale, 1.)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
            else:
                train_transform = transforms.Compose([
                    transforms.RandomResizedCrop(size=224, scale=(args.min_crop_scale, 1.)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                    ], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor(),
                    normalize,
                ])
                    
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        class_index = get_incremental_index(args, session)
        dataset_class = args.Dataset.CIFAR100

        base_aug_mag = 0
        trainset = dataset_class(root=args.dataroot, train=True, download=False,
                                         index=class_index, base_sess=False, keep_all = True, transform = MultiCropTransform(train_transform, 2),
                                         base_aug_mag=base_aug_mag, way=args.way, shot=args.shot)

    if args.dataset == 'cub200':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if args.rand_aug_sup_con:
            train_transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    ImageNetPolicy(),
                    transforms.ToTensor(),
                    normalize
                ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(size=224, scale=(args.min_crop_scale, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=args.prob_color_jitter),  
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                normalize,
            ])
        dataset_class = args.Dataset.CUB200
        trainset = dataset_class(root=args.dataroot, train=True,
                                       index_path=txt_path, transform = MultiCropTransform(train_transform, 2))

    if args.dataset == 'mini_imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if args.rand_aug_sup_con:
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=224, scale=(args.min_crop_scale, 1.)),
                transforms.RandAugment(num_ops = 3),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=224, scale=(args.min_crop_scale, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=args.prob_color_jitter),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                normalize,
            ])
        dataset_class = args.Dataset.MiniImageNet
        trainset = dataset_class(root=args.dataroot, train=True,
                                       index_path=txt_path, transform = MultiCropTransform(train_transform, 2))

    nclass = args.base_class


    # Now for each previous session i.e. session > 0 and session < curr_session
    # load trainset using the index files. And append data and labels from this dataset to the current one
    # Add the ability to choose the number of exemplars from previous sessions
    for inter_ix in range(1, session): # is = intermediate_sessino
        txt_path = "data/index_list/" + args.dataset + "/session_" + str(inter_ix + 1) + '.txt'
        if args.dataset == "cifar100":
            class_index = get_incremental_index(args, inter_ix)
            inter_set = dataset_class(root=args.dataroot, train=True, download=False, index=class_index, base_sess=False,
                                      way=args.way, shot=args.shot)   # Get data from current index
            trainset.data = np.vstack((trainset.data, inter_set.data))
            trainset.targets = np.hstack((trainset.targets, inter_set.targets))
        else:
            inter_set = dataset_class(root=args.dataroot, train=True, index_path=txt_path, base_sess=False)   # Get data from current index

            if args.exemplars_count != args.shot:
                # Exemplar Control: Append the new data from the previous intermeidate sessions to the current dataset
                inter_targets = np.array(inter_set.targets)
                for i in np.unique(inter_targets):
                    ixs = np.where(inter_targets == i)[0]
                    selected_ixs = list(ixs[:args.exemplars_count])
                    for j in selected_ixs:
                        trainset.data.append(inter_set.data[j])
                        trainset.targets.append(inter_set.targets[j])    
            else:
                trainset.data.extend(inter_set.data)
                trainset.targets.extend(inter_set.targets)

    # Append the base classes to the current dataset
    appendKBaseExemplars(trainset, args, nclass)

    if args.batch_size_replay == 0:
        batch_size_new = trainset.__len__()
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_new, shuffle=True,            # <<< TODO: Shuffled. Check if this is problematic
                                                  num_workers=args.num_workers, pin_memory=True, drop_last = args.drop_last_batch)
    else:
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_replay, shuffle=True,
                                                  num_workers=args.num_workers, pin_memory=True, drop_last = args.drop_last_batch)

    return trainset, trainloader


def appendKBaseExemplars(dataset, args, nclass):
    """
        Take only labels from args.base_class and in data self.data append the single exemplar
    """
    if args.dataset == "cifar100":
        # Get dataset indices under base_class
        for i in range(nclass):
            ind_cl = np.where(i == dataset.targets_all)[0]

            # Choose top 5 from ind_cl and append into data_tmp (done to stay consistent across experiments)
            ind_cl = ind_cl[:args.exemplars_count]

            dataset.data = np.vstack((dataset.data, dataset.data_all[ind_cl]))
            dataset.targets = np.hstack((dataset.targets, dataset.targets_all[ind_cl]))
        return

    label2data = {}
    for k,v in dataset.data2label.items():
        if v < nclass:
            if v not in label2data: label2data[v] = []
            label2data[v].append(k)

    # To maintain simplicity and the reduce added complexity we always sample the first K exemplars from the base class.
    # This should ideally not introduce any biases
    data_tmp = []
    targets_tmp = []

    for i in range(nclass):
        for k in range(args.exemplars_count):
            data_tmp.append(label2data[i][k])
            targets_tmp.append(i)

    dataset.data.extend(data_tmp)
    dataset.targets.extend(targets_tmp)

    return data_tmp, targets_tmp
