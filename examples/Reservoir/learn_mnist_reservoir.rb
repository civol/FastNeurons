require_relative '../../lib/fast_neurons'
require_relative '../../lib/mnist_loader'
require 'gnuplot'

puts "Loading images"

# Load MNIST.
mnist = MNISTLoader.new("../../assets/t10k-images-idx3-ubyte.gz", "../../assets/t10k-labels-idx1-ubyte.gz")
images = mnist.load_images
labels = mnist.load_labels

epochs = 100
cxsize = 2
cysize = 2
isize = 784/(cxsize*cysize)
msize = isize*4

puts "Initializing network"
loss = []

# Initialize the reservoir.
reservoir = FastNeurons::Reservoir.new(isize,msize)
reservoir.gamma = 0.8
reservoir.freeze_R

# Initialize the neural network connected to the reservoir.
# nn = FastNeurons::NN.new([784*4,10], [:Tanh], :MeanSquare)
# nn = FastNeurons::NN.new([msize,msize/16,10], [:Sigmoid,:Sigmoid], :MeanSquare)
nn = FastNeurons::NN.new([msize,10], [:Sigmoid], :MeanSquare)
# nn = FastNeurons::NN.new([784,15,10], [:Sigmoid,:Sigmoid], :SquaredError)
# nn = FastNeurons::NN.new([784,15,10], [:Tanh,:Tanh], :MeanSquare)

# Set learning rate.
nn.set_learning_rate(0.01)

# Set mini-batch size.
#nn.set_batch_size(1)

# Set up the parameters to random values.
nn.randomize(:GlorotNormal, :Zeros)

# Load learned network.
#nn.load_network("network.json")

# Normalize pixel values.
# imgs = images.map { |image| mnist.normalize(image,-1.0..1.0).flatten }
imgs = images.map { |image| mnist.normalize(image).flatten }
# Resize images.
simgs = []
imgs.each do |img|
    simg = []
    simgs << simg
    28.times do |j|
        28.times do |i|
            if (j%cysize == 0) && (i%cxsize == 0) then
                simg << img[j*28+i]
            end
        end
    end
end
imgs = simgs

# Normalize the labels
lbs = labels.map { |label| res = [-1.0] * 10 ; res[label] = 1.0 ; res }
# lbs = labels.map { |label| res = [0.0] * 10 ; res[label] = 1.0 ; res }

puts "Runnning..."

# learning
old_success = 0
epochs.times do |epoch|
    count = 0
    success = 0
    # puts "epoch=#{epoch}"
    imgs.each.with_index do |inputs,index|
        # Send the input to the reservoir.
        nR = reservoir.step(inputs)
        # Use the reservoir as input for the NN.
        # nn.input(inputs,lbs[index]) # Input training data and teaching data.
        nn.input(nR / nR.size,lbs[index]) # Input training data and teaching data.
        nn.run(1) # Compute feed forward propagation and backpropagation.

        # mnist.print_ascii(inputs,-1.0..1.0) # Output training data.
        count += 1
        res = nn.get_outputs.to_a.flatten
        if (lbs[index].each_with_index.max[1] == res.each_with_index.max[1]) then
            success += 1
        end
        # puts "idx=#{res.each_with_index.max[1]} exp=#{lbs[index].each_with_index.max[1]}" 
        # puts "R=#{reservoir.R.to_a}"
        # puts "nR=#{nR.to_a}"
        # puts "nR=#{nR.to_a.flatten}"
        # puts "got output=#{nn.get_outputs.to_a.flatten}"
        # puts "expecting =#{lbs[index]}"
        if (count % 100 == 0) then
            puts "epoch=#{epoch} count=#{count} success=#{success} rate=#{success.to_f/count*100}%"
        end
        # puts "exp   = #{lbs[index]}"
        # puts "guess = #{nn.get_outputs.to_a.flatten}"
        # mnist.print_ascii(nn.get_outputs,-1.0..1.0) # Output the output of neural network.
        # nn.compute_loss
        # loss << nn.get_loss
        # nn.initialize_loss
    end
    # Save learning state after each epoch if better success.
    if (success > old_success) then
        puts "Better weights for rate=#{success.to_f/count*100}%"
        # nn.save_network("network_bnn.json") # save learned network
        old_success = success
    end
    # # Shall we regenerate the reservoir?
    # if epoch == 0 && success.to_f/count < 0.5  then
    #     # Yes.
    #     reservoir.generate_W
    #     redo
    # end
end

puts "Understood!"

check_data = []
# confirmation of network
10.times do
    # Sample an input.
    index = rand(imgs.size)
    input_data = imgs[index]
    # Send it to the reservoir.
    nR = reservoir.step(input_data)
    # Use the reservoir as input for the NN.
    nn.input(nR)
    nn.propagate
    puts "exp   = #{lbs[index]}"
    puts "guess = #{nn.get_outputs.to_a.flatten}"
    mnist.print_ascii(nn.get_outputs,-1.0..1.0) # Output the output of neural network.
end


