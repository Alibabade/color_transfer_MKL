require 'cutorch'


function MKL(A,B)
  
  local timer = torch.Timer()
  local N = A:size(2)
  local Da2, Ua = torch.eig(A, 'V')
  --Da2 = torch.diag(Da2)
  Da2 = Da2:select(2,1)
  for i = 1, Da2:size(1) do 
    if Da2[i] < 0 then
       Da2[i] = 0 
    end
  end
  local Da = torch.diag(torch.sqrt(Da2))
  -- C = Da * Ua:t() * B * Ua * Da
  local C = torch.mm(Da, Ua:t())
  C = torch.mm(C,B)
  C = torch.mm(C,Ua)
  C = torch.mm(C,Da) 


  local Dc2, Uc = torch.eig(C, 'V')
  Dc2 = Dc2:select(2,1)
  for i = 1, Dc2:size(1) do 
    if Dc2[i] < 0 then
       Dc2[i] = 0 
    end
  end
  local Dc = torch.diag(torch.sqrt(Dc2))

  local temp = torch.Tensor(N)
  for i = 1, N do 
    temp[i] = 1/torch.diag(Da):select(2,i)
  end 

  local Da_inv = torch.diag(temp)
  local T = torch.mm(Ua, Da_inv) 
  --T = Ua * Da_inv * Uc * Dc * Uc:t() * Da_inv * Ua:t()
  T = torch.mm(T, Uc)
  T = torch.mm(T, Dc)
  T = torch.mm(T, Uc:t())
  T = torch.mm(T, Da_inv)
  T = torch.mm(T, Ua:t())

  print('Time for MKL: ' ..timer:time().real.. 'seconds')
  return T


end
