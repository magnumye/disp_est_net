require 'nn'
require 'cunn'
require 'image'
require 'colormap'

require('createModelBasicSinglescale.lua')
require('createModelBasicMultiscale.lua')
require('createModelSiameseSinglescale.lua')
require('createModelSiameseMultiscale.lua')
require('createModelSinglescaleTest.lua')
require('createModelMultiscaleTest.lua')


local c = require 'trepl.colorize'

require 'opts.lua'
opt = opt_parse(arg)

colormap:setStyle('gray')
colormap:setSteps(512)

cutorch.setDevice(1)
torch.setdefaulttensortype('torch.FloatTensor')



local ldata = torch.Tensor(opt.test_num, opt.num_channel, opt.height, opt.width)
--local rdata = torch.Tensor(opt.test_num, opt.num_channel, opt.height, opt.width)

for i = 1, opt.test_num do
    local l_img, r_img
    l_img = image.load(string.format('%s/image_0/%06d.png', opt.data_root_test, i), opt.num_channel, 'byte'):float()
    --r_img = image.load(string.format('%s/image_1/%06d.png', opt.data_root_test, i), opt.num_channel, 'byte'):float()
    l_img:add(-128.0):div(255.0)
    --r_img:add(-128.0):div(255.0)

    ldata[i]:copy(l_img)
    --rdata[i]:copy(r_img)
end
local intr = torch.FloatTensor(2)
intr[1] = opt.width
intr[2] = opt.height

if opt.model_no == 1 then
    full_model = createModelBasicSinglescale(opt.height, opt.width)
    Model = createModelSinglescaleTest()
    opt.model_param_root = 'trained/BasicSinglescale'
elseif opt.model_no == 2 then
    full_model = createModelBasicMultiscale(opt.height, opt.width)
    Model = createModelMultiscaleTest()
    opt.model_param_root = 'trained/BasicMultiscale'
elseif opt.model_no == 3 then
    full_model = createModelSiameseSinglescale(opt.height, opt.width)
    Model = createModelSinglescaleTest()
    opt.model_param_root = 'trained/SiameseSinglescale'
elseif opt.model_no == 4 then
    full_model = createModelSiameseMultiscale(opt.height, opt.width)
    Model = createModelMultiscaleTest()
    opt.model_param_root = 'trained/SiameseMultiscale'
end


local model_param, model_grad_param = Model:getParameters()
local full_param, _ = full_model:getParameters()
--print(string.format('number of parameters: %d', model_param:nElement()))
--print(string.format('number of parameters: %d', full_param:nElement()))
print(c.blue '==>' .. ' loading parameters')
local param_file = 'param_epoch_50.t7'
local bn_file = 'bn_meanvar_epoch_50.t7'
local params = torch.load(paths.concat(opt.model_param_root, param_file))
assert(params:nElement() == full_param:nElement(), string.format('%s: %d vs %d', 'loading parameters: dimension mismatch.', params:nElement(), model_param:nElement()))
full_param:copy(params)

for i = 1, #full_model.modules do
    --print(full_model.modules[i].__typename)
    local cparam, _ = full_model.modules[i]:getParameters()
    if cparam:nElement() > 100000 then -- larger than a rough number, a bit ad-hoc for now
        --print(cparam:nElement())
        model_param:copy(cparam)
        break
    end

end

-- load BN mean and std
local bn_mean, bn_std = table.unpack(torch.load(paths.concat(opt.model_param_root, bn_file)))

for k, v in pairs(Model:findModules('nn.SpatialBatchNormalization')) do
    v.running_mean:copy(bn_mean[k])
    v.running_var:copy(bn_std[k])
end

Model:cuda()
Model:evaluate()

disps = torch.Tensor(opt.test_num, opt.height, opt.width):fill(0)

function evaluate_disp()
    print(c.blue '==>'..string.format(" disparity estimation for %d images", opt.test_num))
    local inputs_test = ldata:cuda()
 	local n = (#inputs_test)[1]
    assert(math.fmod(n, opt.batch_size_test) == 0, "use opt.batch_size_test to be divided exactly by number of testing samples")

	for i = 1, n, opt.batch_size_test do
		local output_test = Model:forward(inputs_test:narrow(1,i,opt.batch_size_test):cuda())
        if opt.model_no == 2 or opt.model_no == 4 then
            disps:narrow(1,i,opt.batch_size_test):copy(output_test[1]:float())
        else
            disps:narrow(1,i,opt.batch_size_test):copy(output_test:float())
        end

    end

end

local tic = torch.tic()
evaluate_disp()
local time_average = torch.toc(tic)*1000/opt.test_num
print(string.format('%.2f ms per forward pass for image size %d x %d.', time_average, opt.width, opt.height))

disps:resize(opt.test_num, 1, opt.height, opt.width)
for i = 1, opt.test_num do

    local rgbImg = colormap:convert(disps[i])
    local filename = string.format('disp_%06d.png', i)
    image.save(paths.concat(opt.save_test, filename),rgbImg)

end

print(string.format("Finished. Results have been saved to %s/", opt.save_test))
