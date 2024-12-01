import argparse
from dataset import Dataset

def str2bool(v):
    return v.lower() in ("true", "1")
# Parse command line arguments.
parser = argparse.ArgumentParser(description='extract sift.')
parser.add_argument('--raw_data_path', type=str, default='../raw_data/',
  help='raw data path. default:../raw_data/')
parser.add_argument('--dump_dir', type=str, default='../data_dump/',
  help='data dump path. default:../data_dump')
parser.add_argument('--desc_name', type=str, default='sift-2000',
  help='prefix of desc filename, default:sift-2000')
parser.add_argument('--vis_th', type=int, default=50,
  help='visibility threshold')
parser.add_argument('--pair_num', type=int, default=1000,
  help='pair num. 1000 for test seq')
parser.add_argument('--desc_split', type=int, default=4,
  help='sub descriptor parts')
parser.add_argument('--min_matches', type=int, default=0,
  help='minimum number of matches(Exclusive)')
parser.add_argument('--double_desc', type=bool, default=False,
  help='Using double desc')



if __name__ == "__main__":
    config = parser.parse_args()
    # dump yfcc test
    test_seqs = ['buckingham_palace','notre_dame_front_facade','reichstag', 'sacre_coeur']
    # test_seqs = ['buckingham_palace']

    double_descs = ["sift_brisk", "brisk_sift", "orb_brisk", "orb_sift", "double_aliked", "triple_aliked" ]

    config.double_desc = config.desc_name in double_descs
 
    yfcc_te = Dataset(config.raw_data_path+'yfcc100m/', config.dump_dir, 'yfcc-'+config.desc_name+'-test.hdf5', \
        test_seqs, 'test', config.desc_name, \
        config.vis_th, config.pair_num, config.raw_data_path+'pairs/', config)

    # # dump yfcc training seqs
    # with open('yfcc_train.txt','r') as ofp:
    #     train_seqs = ofp.read().split('\n')
    # if len(train_seqs[-1]) == 0:
    #     del train_seqs[-1]
    # print('train seq len '+str(len(train_seqs)))
    # yfcc_tr_va = Dataset(config.raw_data_path+'yfcc100m/', config.dump_dir, 'yfcc-'+config.desc_name+'-val.hdf5', \
    #     train_seqs, 'val', config.desc_name, \
    #     config.vis_th, 100, None)
    # yfcc_tr_tr = Dataset(config.raw_data_path+'yfcc100m/', config.dump_dir, 'yfcc-'+config.desc_name+'-train.hdf5', \
    #     train_seqs, 'train', config.desc_name, \
    #     config.vis_th, 10000, None)
