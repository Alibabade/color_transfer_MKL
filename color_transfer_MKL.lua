require 'torch'
require 'image'
require 'MKL'
require 'reshape'

--[[
This script is implemented for the codes of 
[Pitie07-CVMP]  The Linear Monge-Kantorovith Linear Colour Mapping
                for Example-Based Colour Transfer
                F.Pitie, A. Kokaram (2007)

Editor: Li Wang
Date: June 23 2017


--]]
function main()

  --load images
  local I0 = image.load('scotland_house.png',3 ,'double') --reference image
  local I1 = image.load('scotland_plain.png',3, 'double') --target image
  
  --I0 and I1 must have the same dimensions
  IR = color_transfer_MKL(I0,I1)
  image.save('IR_MKL.png',IR)

end

function color_transfer_MKL(I0, I1)
  
  local timer = torch.Timer()
  local C0,H0,W0 = I0:size(1), I0:size(2), I0:size(3) 
  local X0= reshape(I0,C0,H0,W0)
  local C1,H1,W1 = I1:size(1), I1:size(2), I1:size(3)
  local X1= reshape(I1,C1,H1,W1)

  local A = cov(X0)
  local B = cov(X1)

  local T = MKL(A,B)


  local meanX0 = torch.Tensor(A:size(2))
  local meanX1 = torch.Tensor(B:size(2))
  for i = 1, A:size(2) do 
    meanX0[i] = torch.mean(X0:select(2,i))
    meanX1[i] = torch.mean(X1:select(2,i))
  end

  local mX0 = meanX0:repeatTensor(X0:size(1), 1)
  local mX1 = meanX1:repeatTensor(X0:size(1), 1)
  
  --local XR = (X0 - mX0) * T + mX1
  local XR = torch.add(X0, -1, mX0)
  XR = torch.mm(XR, T)
  XR = torch.add(XR,mX1)

  local IR = unreshape(XR, I0)
  print('Time for all: ' .. timer:time().real.. 'seconds')
  return IR


end


-- this function is implemented for the cov function of matlab
function cov(A)
  local timer = torch.Timer()

  local m,n = A:size(1), A:size(2)
  local d = torch.Tensor(n, m)

  local u = torch.Tensor(n,1)
  for i = 1, n do
    u[i]= torch.mean(A:select(2,i))
  end

  local x_t = A:t()
  local mX0 = u:repeatTensor(1, x_t:size(2))

  d = torch.add(x_t, -1, mX0)

  --local cov = d * d:t() / (m - 1 )
  local cov = torch.mm(d, d:t())
  cov = torch.div(cov, m-1)


  print('Time for cov:' ..timer:time().real.. 'seconds')
  return cov
end


main()
