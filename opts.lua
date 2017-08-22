function opt_parse(arg)
    local cmd = torch.CmdLine()

    cmd:option('-model_no', 4, 'Options: 1(basic single) | 2(basic multi) | 3(siamese single) | 4(siamese multi)')
    cmd:option('-data_root', 'data/daVinci/train', 'Training dataset directory')
    cmd:option('-batch_size', 4, 'mini-batch size')
    cmd:option('-niter', 1250, 'Number of iterations for each epoch')
    cmd:option('-nepoch', 50, 'Number of epochs')
    cmd:option('-epoch_step', 5, 'Learning rate decay per # epochs')
    cmd:option('-tr_num', 5000, 'Number of pairs of images for training')
    cmd:option('-height', 192, 'Image height')
    cmd:option('-width', 384, 'Image width')
    cmd:option('-num_channel', 3, 'Image channels')
    cmd:option('-gpu', 1, 'Use GPU')
    cmd:option('-save', 'models', 'Model output directory')
    cmd:option('-pretrained', 0, 'Use pretrained model parameters')
    cmd:option('-pretrained_root', '../pre_models', 'Pretrained model directory')

    cmd:option('-data_root_test', 'data/daVinci/test', 'testing dataset directory')
    cmd:option('-model_param_root', 'trained/BasicSinglescale', 'model param root for testing')
    cmd:option('-batch_size_test', 4, 'mini-batch size')
    cmd:option('-test_num', 100, 'Number of images for testing')
    cmd:option('-save_test', 'results', 'testing output directory')

    cmd:option('-manualSeed', 123, 'Manually set RNG seed')

    cmd:option('-learningRate', 0.0001, 'Initial learning rate')
    cmd:option('-beta1', 0.9, 'Beta1')
    cmd:option('-beta2', 0.999, 'Beta2')
    cmd:option('-epsilon', 1e-8, 'Epsilon')

    cmd:text()

    local opt = cmd:parse(arg or {})

    if not paths.dirp(opt.save) and not paths.mkdir(opt.save) then
        cmd:error('error: unable to create output directory: ' .. opt.save .. '\n')
    end

    return opt
end