require 'image'
require 'cutorch'
require 'gnuplot'
require 'xlua'

local DataReader = torch.class('DataReader')
function DataReader:__init(data_root, num_channel, height, width, num_tr_img, batch_size, gpu)
    self.nChannel = num_channel
    self.batch_size = batch_size
    self.cuda = gpu or 1
    self.tr_ptr = 0
    self.cur_epoch = 0
    self.ldata = {}
    self.rdata = {}
    self.ldata2 = {}
    self.rdata2 = {}
    self.ldata3 = {}
    self.rdata3 = {}
    self.total_num_img = num_tr_img
    self.num_tr_img = num_tr_img
    self.img_height = height
    self.img_width = width

    self.fids = torch.FloatTensor(torch.FloatStorage(paths.concat(data_root, 'perm.bin')))

    --local total_perm = torch.randperm(self.total_num_img)
    self.tr_perm = self.fids[{{1,num_tr_img}}]

    self.ldata = torch.Tensor(self.total_num_img, self.nChannel, self.img_height, self.img_width)
    self.rdata = torch.Tensor(self.total_num_img, self.nChannel, self.img_height, self.img_width)
    self.ldata2 = torch.Tensor(self.total_num_img, self.nChannel, self.img_height/2, self.img_width/2)
    self.rdata2 = torch.Tensor(self.total_num_img, self.nChannel, self.img_height/2, self.img_width/2)
    self.ldata3 = torch.Tensor(self.total_num_img, self.nChannel, self.img_height/4, self.img_width/4)
    self.rdata3 = torch.Tensor(self.total_num_img, self.nChannel, self.img_height/4, self.img_width/4)

    --print(self.tr_perm)
    for i = 1, self.total_num_img do
        xlua.progress(i, self.total_num_img)
        local l_img, r_img, l_img2, r_img2, l_img3, r_img3
        local fid = self.fids[i]
        l_img = image.load(string.format('%s/image_0/%06d.png', data_root, fid), self.nChannel, 'byte'):float()
        r_img = image.load(string.format('%s/image_1/%06d.png', data_root, fid), self.nChannel, 'byte'):float()

        l_img:add(-128.0):div(255.0)
        r_img:add(-128.0):div(255.0)

        l_img2 = image.scale(l_img, self.img_width/2, self.img_height/2)
        r_img2 = image.scale(r_img, self.img_width/2, self.img_height/2)
        l_img3 = image.scale(l_img, self.img_width/4, self.img_height/4)
        r_img3 = image.scale(r_img, self.img_width/4, self.img_height/4)
        self.ldata[i]:copy(l_img)
        self.rdata[i]:copy(r_img)
        self.ldata2[i]:copy(l_img2)
        self.rdata2[i]:copy(r_img2)
        self.ldata3[i]:copy(l_img3)
        self.rdata3[i]:copy(r_img3)

    end

    -- reserve memory for training batch and load validation set
    self.batch_left = torch.Tensor(self.batch_size, self.nChannel, self.img_height, self.img_width)
    self.batch_right = torch.Tensor(self.batch_size, self.nChannel, self.img_height, self.img_width)
    self.batch_left2 = torch.Tensor(self.batch_size, self.nChannel, self.img_height/2, self.img_width/2)
    self.batch_right2 = torch.Tensor(self.batch_size, self.nChannel, self.img_height/2, self.img_width/2)
    self.batch_left3 = torch.Tensor(self.batch_size, self.nChannel, self.img_height/4, self.img_width/4)
    self.batch_right3 = torch.Tensor(self.batch_size, self.nChannel, self.img_height/4, self.img_width/4)

    self.zero_target = torch.Tensor(self.batch_size, 1, self.img_height, self.img_width):zero()
    self.zero_target2 = torch.Tensor(self.batch_size, 1, self.img_height/2, self.img_width/2):zero()
    self.zero_target3 = torch.Tensor(self.batch_size, 1, self.img_height/4, self.img_width/4):zero()

    print(string.format('Dataset created for training %d images, with batch size %d.', num_tr_img, batch_size))

    collectgarbage()
end

function DataReader:next_batch()
    for i = 1, self.batch_size do
        local idx = self.tr_ptr + i
        if idx > torch.numel(self.tr_perm) then
            idx = 1
            self.tr_ptr = -i + 1
            self.cur_epoch = self.cur_epoch + 1
            print('....Epoch id: ' .. self.cur_epoch .. ' done ......\n')
        end
        local img_id = idx --self.tr_perm[idx]
        self.batch_left[i]:copy(self.ldata[img_id])
        self.batch_right[i]:copy(self.rdata[img_id])

    end
    self.tr_ptr = self.tr_ptr + self.batch_size

    if self.cuda == 1 then
        return self.batch_left:cuda(), self.batch_right:cuda(), self.zero_target:cuda()
    else
        return self.batch_left, self.batch_right, self.zero_target
    end
end

function DataReader:next_batch_multiscale()
    for i = 1, self.batch_size do
        local idx = self.tr_ptr + i
        if idx > torch.numel(self.tr_perm) then
            idx = 1
            self.tr_ptr = -i + 1
            self.cur_epoch = self.cur_epoch + 1
            print('....Epoch id: ' .. self.cur_epoch .. ' done ......\n')
        end
        local img_id = idx --self.tr_perm[idx]
        self.batch_left[i]:copy(self.ldata[img_id])
        self.batch_right[i]:copy(self.rdata[img_id])
        self.batch_left2[i]:copy(self.ldata2[img_id])
        self.batch_right2[i]:copy(self.rdata2[img_id])
        self.batch_left3[i]:copy(self.ldata3[img_id])
        self.batch_right3[i]:copy(self.rdata3[img_id])

    end
    self.tr_ptr = self.tr_ptr + self.batch_size

    if self.cuda == 1 then
        return self.batch_left:cuda(), self.batch_left2:cuda(), self.batch_left3:cuda(),
        self.batch_right:cuda(), self.batch_right2:cuda(), self.batch_right3:cuda(),
        self.zero_target:cuda(), self.zero_target2:cuda(), self.zero_target3:cuda()
    else
        return self.batch_left, self.batch_left2, self.batch_left3,
        self.batch_right, self.batch_right2, self.batch_right3,
        self.zero_target, self.zero_target2, self.zero_target3
    end
end
