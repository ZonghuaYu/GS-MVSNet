import torch,argparse,os,time
from torch.utils.data import DataLoader

import config,logging
from load.evaltanks import LoadDataset
from tools.data_io import save_pfm,write_depth_img


def eval(args):
    os.makedirs(args.output_path, exist_ok=True)

    # creat model
    model = config.model
    model.set_niters(args.num_iters)

    # load breakpoint
    if args.pre_model is not None:
        checkpoint = torch.load(args.pre_model) #map_location=torch.device("cpu")   ,map_location=DEVICE
        model.load_state_dict(checkpoint["model"])  # strict=True

    model.to(config.DEVICE)

    # load dataset
    eval_dataset = LoadDataset(datasetpath=config.datasetpath, scenelist=config.scenelist, nviews=args.num_views)
    eval_databatch = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=1,
                               pin_memory=True,drop_last=False)

    #eval
    # don't use model.eval()
    with torch.no_grad():
        for iteration, data in enumerate(eval_databatch):
            torch.cuda.empty_cache()
            data_batch = {k: v.to(config.DEVICE) for k, v in data.items() if isinstance(v, torch.Tensor)}

            start_time = time.time()
            outputs = model(data_batch["imgs"], data_batch["extrinsics"], data_batch["intrinsics"], data_batch["depth_range"])
            logging.info("batch:"+str(iteration + 1)+ "/"+ str(len(eval_databatch))+" time = {:.3f}".format(time.time() - start_time))

            del data_batch
            # save depth map,confidence map, depth img
            for filename, depth,photometric_confidence in \
                    zip(data["filename"], outputs["depth"], outputs["confidence"]):  #(B,H,W)
                depth_filename = os.path.join(args.output_path, filename.format('depth_est', '.pfm'))
                depthimg_filename = os.path.join(args.output_path, filename.format('depth_est', '.png'))
                confidence_filename = os.path.join(args.output_path, filename.format('confidence', '.pfm'))

                os.makedirs(depth_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(confidence_filename.rsplit('/', 1)[0], exist_ok=True)

                # save depth maps
                save_pfm(depth_filename, depth.cpu())
                write_depth_img(depthimg_filename, depth.cpu().numpy())
                # Save prob maps
                save_pfm(confidence_filename, photometric_confidence.cpu())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tanks eval parameter setting')
    parser.add_argument('-i', '--num_iters', default=5, type=int, help='number of iterations of the network')
    parser.add_argument('-v', '--num_views', default=7, type=int, help='views for input')
    parser.add_argument('-p', '--pre_model', default=None, type=str, help='Pre training model')
    parser.add_argument('-o', '--output_path', default=config.eval_output_path, type=str, help='output folder location')

    args = parser.parse_args()
    logging.info(args)

    eval(args)

