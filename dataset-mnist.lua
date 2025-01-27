require 'torch'
require 'paths'

mnist = {}

mnist.path_dataset = 'mnist.t7'
mnist.path_trainset = paths.concat(mnist.path_dataset, 'train_32x32.t7')
mnist.path_testset = paths.concat(mnist.path_dataset, 'test_32x32.t7')

function mnist.loadTrainSet(maxLoad, geometry)
   local data = mnist.loadDataset(mnist.path_trainset, maxLoad, geometry)
   data:normalizeGlobal(mean, std)
   local data_table = {}
   for i = 1,maxLoad do 
      local data_table_temp = {
             data = data.data[i],
             filename = "",
             labelname = "",
             labels = data.labels[i],
             sets = 1 
         }
      data_table[#data_table + 1] = data_table_temp
   end
   return data_table
end

function mnist.loadTestSet(maxLoad, geometry)
   local data = mnist.loadDataset(mnist.path_testset, maxLoad, geometry)
   data:normalizeGlobal(mean, std)
   local data_table = {}
   for i = 1,maxLoad do 
      local data_table_temp = {
             data = data.data[i],
             filename = "",
             labelname = "",
             labels = data.labels[i],
             sets = 1 
         }
      data_table[#data_table + 1] = data_table_temp
   end
   return data_table
end

function mnist.loadDataset(fileName, maxLoad)

   local f = torch.load(fileName, 'ascii')
   local data = f.data:type(torch.getdefaulttensortype())
   local labels = f.labels

   local nExample = f.data:size(1)
   if maxLoad and maxLoad > 0 and maxLoad < nExample then
      nExample = maxLoad
      print('<mnist> loading only ' .. nExample .. ' examples')
   end
   data = data[{{1,nExample},{},{},{}}]
   labels = labels[{{1,nExample}}]
   print('<mnist> done')

   local dataset = {}
   dataset.data = data
   dataset.labels = labels

   function dataset:normalize(mean_, std_)
      local mean = mean or data:view(data:size(1), -1):mean(1)
      local std = std_ or data:view(data:size(1), -1):std(1, true)
      for i=1,data:size(1) do
         data[i]:add(-mean[1][i])
         if std[1][i] > 0 then
            tensor:select(2, i):mul(1/std[1][i])
         end
      end
      return mean, std
   end

   function dataset:normalizeGlobal(mean_, std_)
      local std = std_ or data:std()
      local mean = mean_ or data:mean()
      data:add(-mean)
      data:mul(1/std)
      return mean, std
   end

   function dataset:size()
      return nExample
   end

   local labelvector = torch.zeros(10)

   setmetatable(dataset, {__index = function(self, index)
			     local input = self.data[index]
			     local class = self.labels[index]
			     local label = labelvector:zero()
			     label[class] = 1
			     local example = {input, label}
                                       return example
   end})

   return dataset
end
