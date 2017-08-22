require 'nn'
local grad = require 'autograd'



local n_s=3
local mu_1 = grad.nn.SpatialAveragePooling(n_s,n_s,1,1)
local mu_2 = grad.nn.SpatialAveragePooling(n_s,n_s,1,1)
local mu_3 = grad.nn.SpatialAveragePooling(n_s,n_s,1,1)
local mu_4 = grad.nn.SpatialAveragePooling(n_s,n_s,1,1)
local mu_5 = grad.nn.SpatialAveragePooling(n_s,n_s,1,1)

local cm_1 =  grad.nn.CMulTable()
local cm_2 =  grad.nn.CMulTable()
local cm_3 =  grad.nn.CMulTable()
local cm_4 =  grad.nn.CMulTable()
local cm_5 =  grad.nn.CMulTable()
local cm_6 =  grad.nn.CMulTable()
local cm_7 =  grad.nn.CMulTable()
local cm_8 =  grad.nn.CMulTable()

local cd_1 =  grad.nn.CDivTable()
local C1 = 6.5025
local C2 = 58.5225


-- loss combining ssim and l1 on a batch
local loss_func = function(output, target)
    local app_loss = 0
    local y = output * 255.0 + 128.0 -- back to 8-bit
    local x = target * 255.0 + 128.0 -- back to 8-bit

    local mu_x = mu_1(x)
    local mu_y = mu_2(y)
    local mu_x_sq = cm_1({ mu_x, mu_x })
    local mu_y_sq = cm_2({ mu_y, mu_y })
    local mu_xy = cm_3({ mu_x, mu_y })
    local X_2 = cm_4({ x, x })
    local Y_2 = cm_5({ y, y })
    local XY = cm_6({ x, y })
    local sigma_x_sq = mu_3(X_2) - mu_x_sq
    local sigma_y_sq = mu_4(Y_2) - mu_y_sq
    local sigma_xy = mu_5(XY) - mu_xy
    local A1 = mu_xy * 2 + C1
    local A2 = sigma_xy * 2 + C2
    local B1 = mu_x_sq + mu_y_sq + C1
    local B2 = sigma_x_sq + sigma_y_sq + C2
    local A = cm_7({ A1, A2 })
    local B = cm_8({ B1, B2 })
    app_loss = (1-torch.mean(cd_1({ A, B })))

	return app_loss
end

local SSIMCriterion1 = grad.nn.AutoCriterion('AutoSSIM')(loss_func)
local muCriterion1 = nn.MultiCriterion():add(SSIMCriterion1, 0.8):add(nn.AbsCriterion(),0.2)

local CombCriterion = nn.ParallelCriterion(false)

CombCriterion:add(muCriterion1):add(nn.AbsCriterion())


return CombCriterion