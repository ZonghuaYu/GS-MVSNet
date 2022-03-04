import config
import torch,argparse,os,logging,time
import torch.optim as optim
from torch.utils.data import DataLoader
from load.traindtu import LoadDataset
from net.loss import Loss


def train(args):
    # creat model,loss,optimizer
    model = config.model
    model.set_niters(args.num_iters)

    loss_criterion = Loss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    # load breakpoint
    start_epoch = config.start_epoch
    if args.pre_model is not None:
        checkpoint = torch.load(args.pre_model, )   # map_location=torch.device("cpu")  map_location=DEVICE
        model.load_state_dict(checkpoint["model"])
        start_epoch = checkpoint["epoch"] + 1

    # load to device
    model.to(config.DEVICE)
    loss_criterion = loss_criterion.to(config.DEVICE)

    # load dataset for train
    train_dataset = LoadDataset(datasetpath=config.train_root_dir, pairpath=config.train_pair_path,
                                scencelist=config.train_label,lighting_label=config.train_lighting_label, nviews=config.train_nviews, )
    train_databatch = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=args.num_works,
                                 drop_last=True, pin_memory=True,)

    # train
    model.train()
    for epoch in range(start_epoch, args.max_epoch+1):
        optimizer.param_groups[0]['lr'] = config.lr * ((1 - (epoch-1)/ args.max_epoch)**config.factor)

        epoch_loss = 0
        for iteration, data_batch in enumerate(train_databatch):
            data_batch = {k: v.to(config.DEVICE) for k, v in data_batch.items() if isinstance(v, torch.Tensor)}
            start_time = time.time()

            outputs = model(data_batch["imgs"], data_batch["extrinsics"], data_batch["intrinsics"], data_batch["depth_range"])
            loss = loss_criterion(outputs["depth"], data_batch["ref_depth"], data_batch["depth_range"],)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_loss = loss.detach().item()
            epoch_loss += current_loss
            logging.info("epoch :"+str(epoch)+ " batch:"+str(iteration + 1)+ "/"+ str(len(train_databatch))
                         + " time{: .3f}".format(time.time() - start_time)+ " loss:"+str(current_loss))

        logging.info("epoch: "+str(epoch)+" loss:"+str(epoch_loss/len(train_databatch)))

        #save epoch loss
        with open(os.path.join(config.pth_path, "epoch_loss.txt"), "a") as f:
            f.write(str(epoch_loss/len(train_databatch)) + "\n")

        if epoch % 1 == 0:
            checkpoint = {'model': model.state_dict(), 'epoch': epoch}
            torch.save(checkpoint, os.path.join(config.pth_path, "model_" + str(epoch) + ".pth"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DTU train parameter setting')
    parser.add_argument('-b', '--batch_size', default=8, type=int, help='train batch size')
    parser.add_argument('-e', '--max_epoch', default=16, type=int, help='the max epoch of train')
    parser.add_argument('-w', '--num_works', default=2, type=int, help='dataset load works')
    parser.add_argument('-i', '--num_iters', default=2, type=int, help='Number of iterations of the network')
    parser.add_argument('-p', '--pre_model', default=None, type=str, help='Pre training model or last model') #os.path.join("pth","model_11.pth")

    args = parser.parse_args()
    logging.info(args)

    train(args)
