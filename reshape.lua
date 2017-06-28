require 'torch'

function reshape(I, C, H, W)

  local C0 = I
  local t = torch.Tensor(C,H*W)
  for i = 1, C do 
    t[i] = C0[i]:t():reshape(H*W,1) 
  end 
  return t:t()

end



function unreshape(XR, I)

  local C,H,W = I:size(1), I:size(2), I:size(3)
  local C0 = XR

  C0 = C0:t()
  local t = torch.Tensor(C,H,W)
  for i = 1, C do
     local t1 = C0[i]:reshape(W,H) 
     t[i] = t1:t()
  end

  return t

end


