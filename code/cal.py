import math

# 计算卷积输出尺寸
def conv2d(input_height, input_width, input_channels, kernel_size, stride, padding, num_filters):
    output_height = (input_height - kernel_size + 2 * padding) // stride + 1
    output_width = (input_width - kernel_size + 2 * padding) // stride + 1
    output_channels = num_filters  # 卷积后输出通道数 = 卷积核的数量
    return output_height, output_width, output_channels

# 计算池化输出尺寸
def pool2d(input_height, input_width, input_channels, pool_size, stride):
    output_height = (input_height - pool_size) // stride + 1
    output_width = (input_width - pool_size) // stride + 1
    output_channels = input_channels  # 池化操作通常不会改变通道数
    return output_height, output_width, output_channels

# 计算反卷积输出尺寸
def deconv2d(input_height, input_width, input_channels, kernel_size, stride, padding, num_filters):
    output_height = (input_height - 1) * stride - 2 * padding + kernel_size
    output_width = (input_width - 1) * stride - 2 * padding + kernel_size
    output_channels = num_filters  # 反卷积后输出通道数 = 反卷积核的数量
    return output_height, output_width, output_channels

# 主程序
def main():
    print("欢迎使用图像尺寸计算工具！")
    
    # 获取输入图像尺寸
    input_height = int(input("请输入输入图像的高度: "))
    input_width = int(input("请输入输入图像的宽度: "))
    input_channels = int(input("请输入输入图像的通道数: "))
    
    while True:
        print("\n当前图像尺寸: 高度 = {} 宽度 = {} 通道数 = {}".format(input_height, input_width, input_channels))
        print("\n请选择操作：")
        print("1. 卷积")
        print("2. 池化")
        print("3. 反卷积")
        print("4. 停止")
        
        choice = input("请输入选择 (1/2/3/4): ")

        if choice == '1':  # 卷积
            kernel_size = int(input("请输入卷积核的大小: "))
            stride = int(input("请输入步幅: "))
            padding = int(input("请输入填充大小: "))
            num_filters = int(input("请输入卷积核的数量: "))
            
            input_height, input_width, input_channels = conv2d(input_height, input_width, input_channels, kernel_size, stride, padding, num_filters)
            print("卷积操作后，输出尺寸: 高度 = {} 宽度 = {} 通道数 = {}".format(input_height, input_width, input_channels))
        
        elif choice == '2':  # 池化
            pool_size = int(input("请输入池化核的大小: "))
            stride = int(input("请输入池化步幅: "))
            
            input_height, input_width, input_channels = pool2d(input_height, input_width, input_channels, pool_size, stride)
            print("池化操作后，输出尺寸: 高度 = {} 宽度 = {} 通道数 = {}".format(input_height, input_width, input_channels))
        
        elif choice == '3':  # 反卷积
            kernel_size = int(input("请输入反卷积核的大小: "))
            stride = int(input("请输入步幅: "))
            padding = int(input("请输入填充大小: "))
            num_filters = int(input("请输入反卷积核的数量: "))
            
            input_height, input_width, input_channels = deconv2d(input_height, input_width, input_channels, kernel_size, stride, padding, num_filters)
            print("反卷积操作后，输出尺寸: 高度 = {} 宽度 = {} 通道数 = {}".format(input_height, input_width, input_channels))
        
        elif choice == '4':  # 停止
            print("程序结束！")
            break
        else:
            print("无效的选择，请重新输入！")

if __name__ == "__main__":
    main()