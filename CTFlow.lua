--------------------------------
--TODO transforms tensor to cuda
--------------------------------
require 'torch'
require '../src/utils'
----------------------------------------------------------------------
----------------------------------------------------------------------

print '==> executing all'

-------------------configuration------------------
ClassNLL = true     -- use classNLL or KL
trainModel = true   -- determine the model whether need to be trained
enableCuda = true   -- use cuda: true | false
toyData = false     -- use toy data to test the train process

if enableCuda then
    print "CUDA enable"
    require 'cunn'
    require 'cutorch'
    --torch.setdefaulttensortype('torch.CudaTensor')
else
    --torch.setdefaulttensortype('torch.FloatTensor')
end

nInput = 1024--*7*7              -- To be modified! Number of all LLC encode, 7*7*64
constraint_constant = 500        -- constant for scalar LLC code

-------------------configuration-------------------
---------------------------------------------------
function subrange(t, first, last)
    local sub = {}
    for i=first,last do
        sub[#sub + 1] = t[i]
    end
    return sub
end

------------------------------
-- toy data from digit dataset
if toyData then 
    dofile 'dataset-mnist.lua'
    classes = {'1','2','3','4','5','6','7','8','9','10'}
    geometry = {32,32}
    nInput = 1024

    nbTrainingPatches = 20
    nbTestingPatches = 10
    -- create training set and normalize
    train_data = mnist.loadTrainSet(nbTrainingPatches, geometry)
    trsize = #train_data

    -- create test set and normalize
    test_data = mnist.loadTrainSet(nbTestingPatches, geometry)
    tesize = #test_data

    constraint_constant = 1
end
-- end
------------------------------

class_num = #classes   -- class number for ClassNLL target init: 2 | 47

dofile 'mlp_model.lua'
dofile 'loss.lua'
dofile 'train.lua'
dofile 'test.lua'

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)
current_confusion_totalValid = 0
current_test_totalValid = 0
old_loss = 1000
current_loss = 0
-- target to optimization
loss_target = 0.01
loss_difference_target = 0.0001
confusion_totalValid_target = 99
test_totalValid = 73            -- state of the art in CVPR16

if trainModel then
    -- optimization
    epoch = 1
    --for i = 1, 10 do
    i = 0
    while true do
        i = i+1
        train()
        print("current_confusion_totalValid: ".. current_confusion_totalValid .. "%")
        print("current_test_totalValid: ".. current_test_totalValid .. "%")
        print("old_loss: ".. old_loss)
        print("current_loss: ".. current_loss)
        print("loss_difference: ".. math.abs(old_loss - current_loss))
        if (i % 10 == 0) then
            --testInTrainData()
            testInTestData()
            if (current_test_totalValid > test_totalValid) then 
                torch.save("winnermodel", model)
                break
            end
        end

        if (math.abs(old_loss - current_loss) < loss_difference_target) and 
            (current_confusion_totalValid > confusion_totalValid_target) then 
            testInTestData()
            print("############## final test ######################")
            torch.save("resultmodel", model)
            break
        end
        old_loss = current_loss
    end
end
