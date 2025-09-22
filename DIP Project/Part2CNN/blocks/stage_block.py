from .basic_block import basic_block

def stage_block(x,filters,num_blocks, downsample):
    for i in range(num_blocks):
        x=basic_block(x,filters,stride=downsample if i==0 else 1)
    return x