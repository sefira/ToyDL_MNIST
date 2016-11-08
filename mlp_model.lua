--------------------------------
--TODO transforms tensor to cuda
--------------------------------

require 'torch'
require 'nn'

----------------------------------------------------------------------

model = nn.Sequential()
model:add(nn.SpatialMaxPooling(7,7))
model:add(nn.MulConstant(constraint_constant))
model:add(nn.View(-1))
model:add(nn.Linear(nInput, 512))
model:add(nn.ReLU())
model:add(nn.Dropout(0.500000))
model:add(nn.Linear(512, 512))
model:add(nn.ReLU())
model:add(nn.Dropout(0.500000))
model:add(nn.Linear(512, class_num))
model:add(nn.LogSoftMax())

-- and move these to the GPU:

if enableCuda then
    model:cuda()
else
    model:float()
end

----------------------------------------------------------------------
print '==> here is the model:'
print(model)
