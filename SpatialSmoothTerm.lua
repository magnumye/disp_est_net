SpatialSmoothTerm, parent = torch.class('nn.SpatialSmoothTerm', 'nn.Sequential')

function SpatialSmoothTerm:__init()
    parent.__init(self)

    local gx = torch.Tensor(3,3):zero()
    gx[2][1] = -1
    gx[2][2] =  0
    gx[2][3] =  1
    --gx = gx:cuda()
    local gradx = nn.SpatialConvolution(1,1,3,3,1,1,1,1)
    gradx.weight:copy(gx)
    gradx.bias:fill(0)
    local gradx_1 = nn.SpatialConvolution(1,1,3,3,1,1,1,1)
    gradx_1.weight:copy(gx)
    gradx_1.bias:fill(0)

    local gy = torch.Tensor(3,3):zero()
    gy[1][2] = -1
    gy[2][2] =  0
    gy[3][2] =  1
    --gy = gy:cuda()
    local grady = nn.SpatialConvolution(1,1,3,3,1,1,1,1)
    grady.weight:copy(gy)
    grady.bias:fill(0)
    local grady_1 = nn.SpatialConvolution(1,1,3,3,1,1,1,1)
    grady_1.weight:copy(gy)
    grady_1.bias:fill(0)

    local branchx_0 = nn.Sequential()
    branchx_0:add(gradx):add(nn.Abs())

    local branchy_0 = nn.Sequential()
    branchy_0:add(grady):add(nn.Abs())

    local branchx_1 = nn.Sequential()
    branchx_1:add(gradx_1):add(nn.Abs()):add(nn.MulConstant(-1)):add(nn.Exp())

    local branchy_1 = nn.Sequential()
    branchy_1:add(grady_1):add(nn.Abs()):add(nn.MulConstant(-1)):add(nn.Exp())

    local paral_x = nn.ParallelTable()
    paral_x:add(branchx_0):add(branchx_1)
    local paral_y = nn.ParallelTable()
    paral_y:add(branchy_0):add(branchy_1)

    local mul_x = nn.Sequential()
    mul_x:add(paral_x):add(nn.CMulTable())
    local mul_y = nn.Sequential()
    mul_y:add(paral_y):add(nn.CMulTable())

    local concat = nn.ConcatTable()
    concat:add(mul_x):add(mul_y)
    --local smoothness = nn.Sequential()
    self:add(concat):add(nn.CAddTable())

end

function SpatialSmoothTerm:updateOutput(input)
    return parent.updateOutput(self, input)
end

function SpatialSmoothTerm:updateGradInput(input, gradOutput)
    return parent.updateGradInput(self, input, gradOutput)
end

function SpatialSmoothTerm:accGradParameters()

end