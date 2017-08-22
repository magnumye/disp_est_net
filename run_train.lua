require 'nn'
require 'cunn'
require 'gvnn'
require 'optim'
grad = require 'autograd'

require('DataReader.lua')

require('createModelBasicSinglescale.lua')
require('createModelBasicMultiscale.lua')
require('createModelSiameseSinglescale.lua')
require('createModelSiameseMultiscale.lua')

local c = require 'trepl.colorize'
require 'opts.lua'
opt = opt_parse(arg)


print(opt)

torch.manualSeed(opt.manualSeed)
cutorch.setDevice(1)
torch.setdefaulttensortype('torch.FloatTensor')

if opt.model_no == 1 then
    model_name = 'Basic Singlescale'
elseif opt.model_no == 2 then
    model_name = 'Basic Multiscale'
elseif opt.model_no == 3 then
    model_name = 'Siamese Singlescale'
elseif opt.model_no == 4 then
    model_name = 'Siamese Multiscale'
end

print(c.blue '==>' .. ' create model: ' .. model_name)
if opt.model_no == 1 then
    Model = createModelBasicSinglescale(opt.height, opt.width)
    criterion = require('bss_criterion')
elseif opt.model_no == 2 then
    Model = createModelBasicMultiscale(opt.height, opt.width)
    criterion = require('bms_criterion')
elseif opt.model_no == 3 then
    Model = createModelSiameseSinglescale(opt.height, opt.width)
    criterion = require('sss_criterion')
elseif opt.model_no == 4 then
    Model = createModelSiameseMultiscale(opt.height, opt.width)
    criterion = require('sms_criterion')
end

print(c.blue '==>' .. ' loading data')
my_dataset = DataReader(opt.data_root, opt.num_channel, opt.height, opt.width, opt.tr_num, opt.batch_size, opt.gpu)

if opt.gpu == 1 then
    Model:cuda()
    criterion:cuda()
end

--print(Model)

optimState = {}
optimiser = optim.adam

parameters, gradParameters = Model:getParameters()
print(string.format('number of parameters: %d', parameters:nElement()))

optimConfig = {
    learningRate = opt.learningRate,
    beta1 = opt.beta1,
    beta2 = opt.beta2,
    epsilon = opt.epsilon
}

if opt.pretrained == 1 then
    local e_id = 50
    print(c.blue '==>' .. ' loading parameters')
    local param_file = string.format('param_epoch_%d.t7', e_id)
    --local bn_file = string.format('bn_meanvar_epoch_%d.t7', e_id)
    local params = torch.load(paths.concat(opt.pretrained_root, param_file))
    assert(params:nElement() == parameters:nElement(), string.format('%s: %d vs %d', 'loading parameters: dimension mismatch.', params:nElement(), parameters:nElement()))
    parameters:copy(params)

    -- load BN mean and std
    --local bn_mean, bn_std = table.unpack(torch.load(paths.concat(opt.pretrained_root, bn_file)))

    --for k, v in pairs(Model:findModules('nn.SpatialBatchNormalization')) do
    --   v.running_mean:copy(bn_mean[k])
    --   v.running_var:copy(bn_std[k])
    --end
end

epoch = epoch or 1

function train_siamese_singlescale()
    Model:training()

    -- learning rate decay
    if epoch == 10 then optimConfig.learningRate = optimConfig.learningRate / 2 end
    if epoch > 10 and (epoch - 10) % opt.epoch_step == 0 then optimConfig.learningRate = optimConfig.learningRate / 2 end

    print(c.blue '==>' .. " Epoch # " .. epoch .. ' [batch_size = ' .. opt.batch_size .. ']')

    local tic = torch.tic()
    local train_loss = 0
    for iter = 1, opt.niter do
        xlua.progress(iter, opt.niter)
        local inputs_L, inputs_R, zero_target = my_dataset:next_batch()

        local inputTable = { inputs_L, inputs_R }
        local targetTable = { inputs_L, inputs_R, zero_target, zero_target, zero_target, zero_target }

        local feval = function(x)
            if x ~= parameters then parameters:copy(x) end
            --gradParameters:zero()
            Model:zeroGradParameters()
            local outputs = Model:forward(inputTable)
            local err = criterion:forward(outputs, targetTable)
            local dparams = criterion:backward(outputs, targetTable)
            Model:backward(inputTable, dparams)
            return err, gradParameters
        end

        local _, loss = optimiser(feval, parameters, optimConfig, optimState)
        train_loss = train_loss + loss[1]
        print(('Train loss ' .. loss[1]))
    end
    print(('Train loss: ' .. c.cyan '%.4f' .. ' \t time: %.2f s\t grad/param norm = %6.4e\t learning rate: %f'):format(train_loss / opt.niter, torch.toc(tic), gradParameters:norm() / parameters:norm(), optimConfig.learningRate))
    trainLogger:add { train_loss / opt.niter, optimConfig.learningRate }
    epoch = epoch + 1
end

function train_siamese_multiscale()
    Model:training()

    -- learning rate decay
    if epoch == 10 then optimConfig.learningRate = optimConfig.learningRate / 2 end
    if epoch > 10 and (epoch - 10) % opt.epoch_step == 0 then optimConfig.learningRate = optimConfig.learningRate / 2 end

    print(c.blue '==>' .. " Epoch # " .. epoch .. ' [batch_size = ' .. opt.batch_size .. ']')

    local tic = torch.tic()
    local train_loss = 0
    for iter = 1, opt.niter do
        xlua.progress(iter, opt.niter)
        local inputs_L1, inputs_L2, _, inputs_R1, inputs_R2, _, zero_target1, zero_target2, _ = my_dataset:next_batch_multiscale()

        local inputTable = { inputs_L1, inputs_L2, inputs_R1, inputs_R2 }
        local targetTable = { inputs_L1, inputs_R1, zero_target1, zero_target1, zero_target1, zero_target1,
        inputs_L2, inputs_R2, zero_target2, zero_target2, zero_target2, zero_target2}

        local feval = function(x)
            if x ~= parameters then parameters:copy(x) end
            --gradParameters:zero()
            Model:zeroGradParameters()
            local outputs = Model:forward(inputTable)
            local err = criterion:forward(outputs, targetTable)
            local dparams = criterion:backward(outputs, targetTable)
            Model:backward(inputTable, dparams)
            return err, gradParameters
        end

        local _, loss = optimiser(feval, parameters, optimConfig, optimState)
        train_loss = train_loss + loss[1]
        print(('Train loss ' .. loss[1]))
    end
    print(('Train loss: ' .. c.cyan '%.4f' .. ' \t time: %.2f s\t grad/param norm = %6.4e\t learning rate: %f'):format(train_loss / opt.niter, torch.toc(tic), gradParameters:norm() / parameters:norm(), optimConfig.learningRate))
    trainLogger:add { train_loss / opt.niter, optimConfig.learningRate }
    epoch = epoch + 1
end

function train_basic_singlescale()
     Model:training()

    -- learning rate decay
    if epoch == 10 then optimConfig.learningRate = optimConfig.learningRate / 2 end
    if epoch > 10 and (epoch - 10) % opt.epoch_step == 0 then optimConfig.learningRate = optimConfig.learningRate / 2 end

    print(c.blue '==>' .. " Epoch # " .. epoch .. ' [batch_size = ' .. opt.batch_size .. ']')

    local tic = torch.tic()
    local train_loss = 0
    for iter = 1, opt.niter do
        xlua.progress(iter, opt.niter)
        local inputs_L, inputs_R, zero_target = my_dataset:next_batch()

        local inputTable = { inputs_L, inputs_R }
        local targetTable = { inputs_L, zero_target }

        local feval = function(x)
            if x ~= parameters then parameters:copy(x) end
            --gradParameters:zero()
            Model:zeroGradParameters()
            local outputs = Model:forward(inputTable)
            local err = criterion:forward(outputs, targetTable)
            local dparams = criterion:backward(outputs, targetTable)
            Model:backward(inputTable, dparams)
            return err, gradParameters
        end

        local _, loss = optimiser(feval, parameters, optimConfig, optimState)
        train_loss = train_loss + loss[1]
        print(('Train loss ' .. loss[1]))
    end
    print(('Train loss: ' .. c.cyan '%.4f' .. ' \t time: %.2f s\t grad/param norm = %6.4e\t learning rate: %f'):format(train_loss / opt.niter, torch.toc(tic), gradParameters:norm() / parameters:norm(), optimConfig.learningRate))
    trainLogger:add { train_loss / opt.niter, optimConfig.learningRate }
    epoch = epoch + 1
end

function train_basic_multiscale()
     Model:training()

    -- learning rate decay
    if epoch == 10 then optimConfig.learningRate = optimConfig.learningRate / 2 end
    if epoch > 10 and (epoch - 10) % opt.epoch_step == 0 then optimConfig.learningRate = optimConfig.learningRate / 2 end

    print(c.blue '==>' .. " Epoch # " .. epoch .. ' [batch_size = ' .. opt.batch_size .. ']')

    local tic = torch.tic()
    local train_loss = 0
    for iter = 1, opt.niter do
        xlua.progress(iter, opt.niter)
        local inputs_L1, inputs_L2, _, inputs_R1, inputs_R2, _, zero_target1, zero_target2, _ = my_dataset:next_batch_multiscale()

        local inputTable = { inputs_L1, inputs_L2, inputs_R1, inputs_R2 }
        local targetTable = { inputs_L1, zero_target1, inputs_L2, zero_target2 }

        local feval = function(x)
            if x ~= parameters then parameters:copy(x) end
            --gradParameters:zero()
            Model:zeroGradParameters()
            local outputs = Model:forward(inputTable)
            local err = criterion:forward(outputs, targetTable)
            local dparams = criterion:backward(outputs, targetTable)
            Model:backward(inputTable, dparams)
            return err, gradParameters
        end

        local _, loss = optimiser(feval, parameters, optimConfig, optimState)
        train_loss = train_loss + loss[1]
        print(('Train loss ' .. loss[1]))
    end
    print(('Train loss: ' .. c.cyan '%.4f' .. ' \t time: %.2f s\t grad/param norm = %6.4e\t learning rate: %f'):format(train_loss / opt.niter, torch.toc(tic), gradParameters:norm() / parameters:norm(), optimConfig.learningRate))
    trainLogger:add { train_loss / opt.niter, optimConfig.learningRate }
    epoch = epoch + 1
end

print('Model will be saved at ' .. opt.save)
paths.mkdir(opt.save)
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
trainLogger:setNames { 'Train Loss', 'Learning Rate' }

function logging()
    -- save model parameters every # epochs
    if epoch % 2 == 0 or epoch == opt.nepoch then
        local filename = paths.concat(opt.save, string.format('param_epoch_%d.t7', epoch))
        print('==> saving parameters to ' .. filename)
        torch.save(filename, parameters)

        -- save bn statistics from training set
        filename = paths.concat(opt.save, string.format('bn_meanvar_epoch_%d.t7', epoch))
        print('==> saving bn mean var to ' .. filename)
        local bn_mean = {}
        local bn_var = {}
        for k, v in pairs(Model:findModules('nn.SpatialBatchNormalization')) do
            bn_mean[k] = v.running_mean
            bn_var[k] = v.running_var
        end
        if #bn_mean > 0 then torch.save(filename, { bn_mean, bn_var }) end

    end
end

while epoch < opt.nepoch do

    if opt.model_no == 4 then
        train_siamese_multiscale()
    elseif opt.model_no == 3 then
        train_siamese_singlescale()
    elseif opt.model_no == 2 then
        train_basic_multiscale()
    elseif opt.model_no == 1 then
        train_basic_singlescale()
    end
    --evaluate()
    logging()

    if epoch % 2 == 0 then collectgarbage() end
end

