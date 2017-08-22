require 'nngraph'

gModule = torch.getmetatable('nn.gModule')

function gModule:share(gModuleToShare, ...)
    for indexNode, node in ipairs(self.forwardnodes) do
        if node.data.module then
            node.data.module:share(gModuleToShare.forwardnodes[indexNode].data.module, ...)
        end
    end
end
