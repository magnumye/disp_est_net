require 'nn'
require 'cunn'
require 'gvnn'
require 'nngraph'


local BN = nn.SpatialBatchNormalization
local MP = nn.SpatialMaxPooling
local MU = nn.SpatialMaxUnpooling
local RL = nn.ReLU
local conv = nn.SpatialConvolution
local deconv = nn.SpatialFullConvolution
local MT = nn.MapTable


local pool = {}
pool[1] = MP(2, 2, 2, 2, 0, 0)
pool[2] = MP(2, 2, 2, 2, 0, 0)
pool[3] = MP(2, 2, 2, 2, 0, 0)
pool[4] = MP(2, 2, 2, 2, 0, 0)
pool[5] = MP(2, 2, 2, 2, 0, 0)

function createModelBasicSinglescale(h, w)
    local height, width
    height = h
    width = w
    local input_imgs_L = nn.Identity()()
    local input_imgs_R = nn.Identity()()

    local function createAutoEncoder()
        local input_data = nn.Identity()()
        local cnv_1_1 = RL(true)(BN(64, 1e-3)(conv(3, 64, 3, 3, 1, 1, 1, 1)(input_data)))
        local cnv_1_2 = RL(true)(BN(64, 1e-3)(conv(64, 64, 3, 3, 1, 1, 1, 1)(cnv_1_1)))

        -- -> size/2
        local pool1 = pool[1](cnv_1_2)
        local cnv_2_1 = RL(true)(BN(128, 1e-3)(conv(64, 128, 3, 3, 1, 1, 1, 1)(pool1)))
        local cnv_2_2 = RL(true)(BN(128, 1e-3)(conv(128, 128, 3, 3, 1, 1, 1, 1)(cnv_2_1)))

        -- -> size/4
        local pool2 = pool[2](cnv_2_2)
        local cnv_3_1 = RL(true)(BN(256, 1e-3)(conv(128, 256, 3, 3, 1, 1, 1, 1)(pool2)))
        local cnv_3_2 = RL(true)(BN(256, 1e-3)(conv(256, 256, 3, 3, 1, 1, 1, 1)(cnv_3_1)))
        local cnv_3_3 = RL(true)(BN(256, 1e-3)(conv(256, 256, 3, 3, 1, 1, 1, 1)(cnv_3_2)))

        -- -> size/8
        local pool3 = pool[3](cnv_3_3)
        local cnv_4_1 = RL(true)(BN(512, 1e-3)(conv(256, 512, 3, 3, 1, 1, 1, 1)(pool3)))
        local cnv_4_2 = RL(true)(BN(512, 1e-3)(conv(512, 512, 3, 3, 1, 1, 1, 1)(cnv_4_1)))
        local cnv_4_3 = RL(true)(BN(512, 1e-3)(conv(512, 512, 3, 3, 1, 1, 1, 1)(cnv_4_2)))

        -- -> size/16
        local pool4 = pool[4](cnv_4_3)
        local decnv_5 = RL(true)(BN(512, 1e-3)(deconv(512, 512, 3, 3, 1, 1, 1, 1)(pool4)))

        -- -> size/8
        local unpool4 = MU(pool[4])(decnv_5)
        local join_4 = nn.JoinTable(2)({unpool4, cnv_4_3})
        local decnv_4_1 = RL(true)(BN(512, 1e-3)(deconv(1024, 512, 3, 3, 1, 1, 1, 1)(join_4)))
        local decnv_4_2 = RL(true)(BN(512, 1e-3)(deconv(512, 512, 3, 3, 1, 1, 1, 1)(decnv_4_1)))
        local decnv_4_3 = RL(true)(BN(256, 1e-3)(deconv(512, 256, 3, 3, 1, 1, 1, 1)(decnv_4_2)))

        -- -> size/4
        local unpool3 = MU(pool[3])(decnv_4_3)
        local join_3 = nn.JoinTable(2)({unpool3, cnv_3_3})
        local decnv_3_1 = RL(true)(BN(256, 1e-3)(deconv(512, 256, 3, 3, 1, 1, 1, 1)(join_3)))
        local decnv_3_2 = RL(true)(BN(256, 1e-3)(deconv(256, 256, 3, 3, 1, 1, 1, 1)(decnv_3_1)))
        local decnv_3_3 = RL(true)(BN(128, 1e-3)(deconv(256, 128, 3, 3, 1, 1, 1, 1)(decnv_3_2)))

        -- -> size/2
        local unpool2 = MU(pool[2])(decnv_3_3)
        local join_2 = nn.JoinTable(2)({unpool2, cnv_2_2})
        local decnv_2_1 = RL(true)(BN(128, 1e-3)(deconv(256, 128, 3, 3, 1, 1, 1, 1)(join_2)))
        local decnv_2_2 = RL(true)(BN(64, 1e-3)(deconv(128, 64, 3, 3, 1, 1, 1, 1)(decnv_2_1)))

        -- -> size/1
        local unpool1 = MU(pool[1])(decnv_2_2)
        local join_1 = nn.JoinTable(2)({unpool1, cnv_1_2})
        local decnv_1_1 = RL(true)(BN(64, 1e-3)(deconv(128, 64, 3, 3, 1, 1, 1, 1)(join_1)))
        local decnv_1_2 = RL(true)(BN(3, 1e-3)(deconv(64, 3, 3, 3, 1, 1, 1, 1)(decnv_1_1)))

        local output_o = conv(3, 1, 3, 3, 1, 1, 1, 1)(decnv_1_2)

        local disp = nn.Sigmoid()(output_o)

        return nn.gModule({input_data}, {disp})

    end

    local disp_net = createAutoEncoder()

    -- init weights
    local method = 'xavier'
    local disp_net_L = require('weight_init')(disp_net, method)


    -- concatenate disparity and transformation
    local disp_L = disp_net_L(input_imgs_L)
    local norm_disp_L = nn.MulConstant(-1)(disp_L)

    -- obtain a sampling grid via STN
    local disp_grid_L = nn.ReverseXYOrder()(nn.Disparity1DBHWD(height,width)(nn.Transpose({ 2, 3 }, { 3, 4 })(norm_disp_L)))

    -- transpose RGB images (right of stereo) to BHWD
    local tranpos_r_net = nn.Transpose({ 2, 3 }, { 3, 4 })(nn.Identity()(input_imgs_R))

    -- concatenate data and STN
    local concat_L = {tranpos_r_net, disp_grid_L}



    -- reconstruct the left/right image using the right/left
    local output_L = nn.Transpose({ 3, 4 }, { 2, 3 })(nn.BilinearSamplerBHWD()(concat_L)) -- Back to BDHW

    -- regulisation term on edge-aware smoothness
    local input_imgs_L_gray = nn.View(-1,1,height,width)(nn.Mean(2)(input_imgs_L))
    local smoothness_L = nn.SpatialSmoothTerm()({disp_L, input_imgs_L_gray})


    local disp_est_net = nn.gModule({ input_imgs_L, input_imgs_R }, { output_L, smoothness_L })


    return disp_est_net
end
