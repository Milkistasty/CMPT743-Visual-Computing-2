import torch
import numpy as np
import torch.utils.data as Data
from torch.autograd import Variable
from models.PoseNet import PoseNet
from data.DataSource import *
from optparse import OptionParser


directory = 'data/datasets/KingsCollege/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(torch.cuda.is_available())

def get_accuracy(pred_xyz, pred_wpqr, poses_gt):
    pose_xyz = poses_gt[0:3]
    pose_wpqr = poses_gt[3:]

    # Calculate position and rotation error
    q1 = pose_wpqr / np.linalg.norm(pose_wpqr)
    q2 = pred_wpqr / np.linalg.norm(pred_wpqr)
    d = abs(np.sum(np.multiply(q1, q2)))
    theta = 2 * np.arccos(d) * 180 / np.pi
    error_x = np.linalg.norm(pose_xyz - pred_xyz)

    return error_x, theta

def main(epoch, data_dir):
    batch_size = 75
    # test dataset and test loader
    datasource = DataSource(directory, train=False)
    test_loader = Data.DataLoader(dataset=datasource, batch_size=batch_size, shuffle=True)
    results = np.zeros((len(test_loader.dataset), 2))

    # load model
    posenet = PoseNet(load_weights=False).to(device)

    save_filename = 'epoch_{}.pth'.format(str(epoch).zfill(5))
    save_path = os.path.join('checkpoints', save_filename)
    posenet.load_state_dict(torch.load(save_path, map_location='cpu'))
    print("Checkpoint {} loaded!".format(save_filename))


    with torch.no_grad():
        posenet.eval()
        for step, (images, poses) in enumerate(test_loader):
            b_images = Variable(images).to(device)
            poses[0] = np.array(poses[0])
            poses[1] = np.array(poses[1])
            poses[2] = np.array(poses[2])
            poses[3] = np.array(poses[3])
            poses[4] = np.array(poses[4])
            poses[5] = np.array(poses[4])
            poses[6] = np.array(poses[5])
            poses = np.transpose(poses)
            b_poses = Variable(torch.Tensor(poses)).to(device)

            p_xyz, p_wpqr = posenet(b_images)

            p_xyz_np = p_xyz.cpu().numpy()
            p_wpqr_np = p_wpqr.cpu().numpy()

            for i in range(b_poses.shape[0]):
                print("{}".format(step*batch_size+i))
                print("GT\t| xyz: {}\twpqr: {}".format(poses[i,:3], poses[i, 3:]))
                print("PRED\t| xyz: {}\twpqr: {}".format(p_xyz_np[i], p_wpqr_np[i]))

                pos_error, ori_error = get_accuracy(p_xyz_np[i], p_wpqr_np[i], poses[i])
                results[step*batch_size+i, :] = [pos_error, ori_error]
                print("ACC\t| pos: {} m \tori: {} degrees".format(pos_error, ori_error))

        median_result = np.median(results, axis=0)
        print("-----------------------------")
        print("Median position error: {} m \t Median orientation error: {} degrees".format(median_result[0], median_result[1]))


def get_args():
    parser = OptionParser()
    parser.add_option('--epoch', default=181, type='int')
    parser.add_option('--data_dir', default='data/datasets/KingsCollege/')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()
    main(epoch=args.epoch, data_dir=args.data_dir)
