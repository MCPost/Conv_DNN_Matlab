function [Weights, Biases, Perform_meas] = ConvDNN_train(Input, Target, Layer, lrate, nEpochs, Batchsize, varargin)

%ConvDNN_train Train a Convolutional Deep Neural Network on some Input data
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% SYNTAX:
% [Weights, Biases, Perform_meas] = ConvDNN_train(Input, Target, Layer, lrate, nEpochs, Batchsize, varargin)
%
% INPUTS:
%   Input   Three dimensional (RxCxN) matrix of input images. R and C are
%           the row and columns of the images, while N is the number of all
%           images used. (Example: 28x28x60000 training images)
%   Target  Nx1 vector with containing the true label of each input image
%           as an integer indicating the target neuron the images category
%           belongs to.
%   Layer   Cell structure containg the build-up of the network. Each row 
%           is a seperate layer and has to have two entries. The name and 
%           the parameters of the layer. Possible values are:
%           - 'Conv': convolutional layer; needs as second input a cell
%                     array with two entries. First, a vector with three
%                     entries k, l, f. k and l are the row and column
%                     dimensions of the feature weights and f is the number
%                     of feature weights used. Second, a two entry vector
%                     with the first entry indicating the stride of the
%                     feature weight and the second entry being the padding
%                     of the input.
%           - 'Pool': max pooling layer; needs as second input an array
%                     with two entries indicating the row and column
%                     dimensions of the pooling matrix.
%           - 'Full': fully connected layer; needs an integer as second
%                     input indicating the number of neurons in this layer.
%   lrate   Learning rate; can be either a number or a vector with one
%           learning rate number for each training epoch.
%   nEpochs   Number of training Epochs; has to be an integer value bigger
%           than 0.
%   Batchsize  Size of Batches used during training; has to be an integer 
%           value between 1 and N. If Batchsize does not divide N equally,
%           not all Input images are used.
% 
% OUTPUT:
%   Weights   Cell array containing the weights for each layer in the first
%           row and additional functions important for forward and backward
%           processing. Note that the weights for Pooling layers are not 
%           weights, but saved filter maps necessary for efficient 
%           computation.
%   Biases  Cell array containing the biases for each layer.
%   Perform_meas  nEpochsx2 vector containing the classification accuracy
%           and the cost of the network on the test data (see optional 
%           inputs)
%
% Optional inputs:
%   Neuron_fct   character string or 1xnEpoch cell array of character
%           strings defining the output activation functions of each layer.
%           Possible values are:
%           - 'sigmoid': sigmoid of logistic function
%           - 'tanh': tanh function
%           - 'relu': rectified linear unit funtion (default)
%           - 'leakyrelu': leaky rectified linear unit funtion
%           - 'softplus': softplus funtion
%   softmax   boolean; if true output layer activation is softmax, false
%           (default)
%   cost    character string defining the cost function. Possible values:
%           - 'quad': quadratic cost function (default)
%           - 'CE': cross-entropy funtion
%   testset   cell array with a Input and Target data set as testset. If
%           missing, Input and Target are used as testset.
%   init    integer or cell array indicating the initialization variance or
%           initialization variance and function used to initialize Weights
%           and Biases (default: init = 0.01). The second entry has to be
%           either a character string 'gauss' for standard normal or 'uni'
%           for uniform distribution or an anonymous function handle with
%           one input values x.
%   drop    number between 0 and 1 indicating how much drop regularization
%           is applied to fully connected layers. 0 means no connection, 1
%           means no dropout at all (default: 0.5).
%   verbose   integer indicating how much feedback the function outputs
%           during learning. Possible values:
%           - 0: no status outputs at all
%           - 1: status of training after each 100 Batches and test data
%                printed to the console (default)
%           - 2: additionally a waitbar shows the status of batch learning
%                (slow)
%           - 3: additionally plots performance measures evolving over
%                training epochs
%   PerfPlot   character string indicating which performance measure is
%           plotted if verbose = 3. Can be either 'acc' for Accuracy or
%           'cost' for Cost.
%
%
% Author: Christopher Postzich, April 2020
%
% This function makes use of the convolve2 function
% Copyright David Young, Feb 2002 
% https://de.mathworks.com/matlabcentral/fileexchange/22619-fast-2-d-convolution
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
varargin_list = reshape(varargin,2,[])';
if(~isempty(varargin_list))
    if(sum(strcmp(varargin_list(:,1),'Neuron_fct')) == 1)
        Neuron_fct = varargin_list{strcmp(varargin_list(:,1),'Neuron_fct'),2};
        if(~iscell(Neuron_fct))
            Neuron_fct = {Neuron_fct};
        elseif(~(length(Neuron_fct) == 1 || length(Neuron_fct) == length(Layer)+1))
            error('Neuron_fct has to have one entry or as much as there are (HiddenLay + 1)!')
        end
    end
    if(sum(strcmp(varargin_list(:,1),'softmax')) == 1)
        softmax = true;
    end
    if(sum(strcmp(varargin_list(:,1),'cost')) == 1)
        cost = varargin_list{strcmp(varargin_list(:,1),'cost'),2};
    end
    if(sum(strcmp(varargin_list(:,1),'testset')) == 1)
        Input_test = varargin_list{strcmp(varargin_list(:,1),'testset'),2}{1};
        Target_test = varargin_list{strcmp(varargin_list(:,1),'testset'),2}{2};
    end
    if(sum(strcmp(varargin_list(:,1),'PerfPlot')) == 1)
        PerfPlot = varargin_list{strcmp(varargin_list(:,1),'PerfPlot'),2};
    end
    if(sum(strcmp(varargin_list(:,1),'init')) == 1)
        init = varargin_list{strcmp(varargin_list(:,1),'init'),2};
    end
    if(sum(strcmp(varargin_list(:,1),'drop')) == 1)
        drop = varargin_list{strcmp(varargin_list(:,1),'drop'),2};
    end
    if(sum(strcmp(varargin_list(:,1),'verbose')) == 1)
        verbose = varargin_list{strcmp(varargin_list(:,1),'verbose'),2};
    end
end


if(~exist('Neuron_fct','var'))
    Neuron_fct = {'sigmoid'};
end
if(~exist('softmax','var'))
    softmax = false;
end
if(~exist('cost','var'))
    cost = 'quad';
end
if(~exist('PerfPlot','var'))
    PerfPlot = 'Acc';
end
if(~exist('init','var'))
    init = 0.01;
end
if(~exist('drop','var'))
    drop = 0.5;
end
if(~exist('Input_test','var') || ~exist('Target_test','var'))
    Input_test = Input;
    Target_test = Target;
end
if(~exist('verbose','var'))
    verbose = 1;
end


if(length(lrate) == 1)
    lrate = lrate*ones(1,nEpochs);
end
if(length(Neuron_fct) == 1)
    Neuron_fct = repmat(Neuron_fct,1,size(Layer,1)+1);
end

for lyr = 1:length(Neuron_fct)
    if(strcmp(Neuron_fct{1,lyr}, 'sigmoid'))        % Sigmoid or logistic function
        Neuron_fct{2,lyr} = @(x) 1./(1 + exp(-1.*x));
        Neuron_fct{3,lyr} = @(x) (1./(1 + exp(-1.*x))).*(1 - (1./(1 + exp(-1.*x))));
    elseif(strcmp(Neuron_fct{1,lyr}, 'tanh'))       % Sigmoid or logistic function
        Neuron_fct{2,lyr} = @(x) tanh(x);
        Neuron_fct{3,lyr} = @(x) 1 - tanh(x).^2;
    elseif(strcmp(Neuron_fct{1,lyr}, 'relu'))       % Rectified Linear Unit (ReLU)
        Neuron_fct{2,lyr} = @(x) x.*((sign(x)+1) == 2);
        Neuron_fct{3,lyr} = @(x) double((sign(x)+1) == 2);
    elseif(strcmp(Neuron_fct{1,lyr}, 'leakyrelu'))  % leaky Rectified Linear Unit (ReLU)
        Neuron_fct{2,lyr} = @(x) x.*(((sign(x)+1) == 2) + .1.*((sign(x)-1) == -2));
        Neuron_fct{3,lyr} = @(x) (((sign(x)+1) == 2) - .1.*((sign(x)-1) == -2));
    elseif(strcmp(Neuron_fct{1,lyr}, 'softplus'))   % softplus
        Neuron_fct{2,lyr} = @(x) log(1 + exp(x));
        Neuron_fct{3,lyr} = @(x) 1./(1 + exp(-1.*x));
    else
        error('Neuron_fct must be called sigmoid, tanh, relu, leakyrelu, softplus!')
    end
    
end

Cost_fct = cell(2,1);
if(strcmp(cost,'quad'))
    Cost_fct{1,1} = @(a, y, z) mean(0.5.*(a - y).^2,1);
    Cost_fct{2,1} = @(a, y, z) (a - y).*z;
elseif(strcmp(cost,'CE'))
    Cost_fct{1,1} = @(a, y, z) -mean(y.*log(a) - (1-y).*log(1-a),1);
    Cost_fct{2,1} = @(a, y, z) (a - y);
else
    error('cost must be called quad (quadratic) or CE (cross-entropy)!')
end

if(strcmp(PerfPlot,'acc'))
    ylab_plot = 'Accuracy (%)';
elseif(strcmp(PerfPlot,'cost'))
    ylab_plot = 'Cost';
else
    error('PerfPlot must be called acc (Accuracy) or cost (Cost)!')
end

if(iscell(init) || length(init) == 1)
    if(iscell(init))
        if(length(init) > 1)
            std_init = init{1};
            if(isa(init{2},'function_handle'))
                init_fct = init{2};
            elseif(strcmp(init{2},'gauss'))
                init_fct = @(x) randn(x);
            elseif(strcmp(init{2},'uni'))
                init_fct = @(x) rand(x);
            else
                error('Second entry of init has to be either a string (options: "uni" for uniform distr, "gauss" for stand norm distr) or function_handle to an anonymous initialization function!')
            end
        else
            std_init = init{1};
            init_fct = @(x) randn(x);
        end
    else
        std_init = init;
        init_fct = @(x) randn(x);
    end
else
    error('init has to be a cell with one (std value) or two (std value, anonymous initialization function)!')
end


% Build Network
N_Layers = {'Input'};
Layer = [{'Input',[size(Input,1) size(Input,2)]}; Layer];
Ntw = {sprintf('Input@(%ix%i) --> ', Layer{1,2})};
Output = cell(2,size(Layer,1)+1);
Output{1,1} = zeros(size(Input,1),size(Input,2),Batchsize);
Output{2,1} = zeros(size(Input,1),size(Input,2),Batchsize);
Weights = cell(6,size(Layer,1));
Biases = cell(1,size(Layer,1));
for i = 2:size(Layer,1)
    if(strcmp(Layer{i,1},'Conv'))
        N_Layers{1,i} = 'Conv';
        Ntw{i,1} = sprintf('Conv@(%ix%ix%i) --> ',Layer{i,2}{1});
        Weights{1,i-1} = std_init.*init_fct(Layer{i,2}{1});
        Weights{2,i-1} = Layer{i,2}{1};
        if(length(Layer{i,2}) == 2)
            Weights{3,i-1} = Layer{i,2}{2}(1);
            Weights{4,i-1} = Layer{i,2}{2}(2);
        else
            Weights{3,i-1} = 1;
            Weights{4,i-1} = 0;
        end
        Weights{5,i-1} = [];
        Weights{6,i-1} = @(x,y) x;
        Weights{7,i-1} = @(x,y) convolve2(x,y);
        Weights{8,i-1} = @(x,y) x.*y;
        Weights{9,i-1} = @(x,y,z) x(:,:,y);
        Biases{1,i-1} = std_init.*init_fct([1 1 Layer{i,2}{1}(3)]);
        Output{1,i} = zeros(1 + (size(Output{1,i-1},1) + 2*Weights{4,i-1} - Layer{i,2}{1}(1))/Weights{3,i-1}, 1 + (size(Output{1,i-1},2) + 2*Weights{4,i-1} - Layer{i,2}{1}(2))/Weights{3,i-1}, Layer{i,2}{1}(3),Batchsize);
        Output{2,i} = zeros(1 + (size(Output{2,i-1},1) + 2*Weights{4,i-1} - Layer{i,2}{1}(1))/Weights{3,i-1}, 1 + (size(Output{2,i-1},2) + 2*Weights{4,i-1} - Layer{i,2}{1}(2))/Weights{3,i-1}, Layer{i,2}{1}(3),Batchsize);
    elseif(strcmp(Layer{i,1},'Pool'))
        N_Layers{1,i} = 'Pool';
        Ntw{i,1} = sprintf('Pool@(%ix%i) --> ',Layer{i,2}{2});
        Weights{2,i-1} = [Layer{i,2}{2} 1];
        Weights{3,i-1} = Layer{i,2}{2};
        Weights{4,i-1} = 0;
        if(strcmp(Layer{i,2}{1},'max'))
            Weights{5,i-1} = [];
            Weights{6,i-1} = @(x,y) kron(x,y);
            Weights{7,i-1} = @(x,y) x.*y;
            Weights{8,i-1} = @(x,y) x;
            Weights{9,i-1} = @(x,y,z) x(:,:,y,z);
        end
        Biases{1,i-1} = 0.*init_fct([1 1 1]);
        Output{1,i} = zeros(floor(size(Output{1,i-1},1)/Weights{2,i-1}(1)), floor(size(Output{1,i-1},2)/Weights{2,i-1}(2)), size(Output{1,i-1},3), Batchsize);
        Output{2,i} = zeros(floor(size(Output{2,i-1},1)/Weights{2,i-1}(1)), floor(size(Output{2,i-1},2)/Weights{2,i-1}(2)), size(Output{2,i-1},3), Batchsize);
    elseif(strcmp(Layer{i,1},'Full'))
        N_Layers{1,i} = 'Full';
        Ntw{i,1} = sprintf('Full@(%i) --> ',Layer{i,2});
        if(regexp(Layer{i-1,1},'Input|Conv|Pool'))
            Weights{1,i-1} = std_init.*init_fct([Layer{i,2} numel(Output{1,i-1})/size(Output{1,i-1},ndims(Output{1,i-1}))]);
            Weights{5,i-1} = @(x) reshape(x, [numel(x)/size(x,4) size(x,4)]);
            Weights{6,i-1} = @(x,y) reshape(x, size(y));
        else
            Weights{1,i-1} = std_init.*init_fct([Layer{i,2} Layer{i-1,2}]);
            Weights{5,i-1} = @(x) x;
            Weights{6,i-1} = @(x,y) x;
        end
        Weights{2,i-1} = [1 1 1];
        Biases{1,i-1} = std_init.*init_fct([Layer{i,2} 1]);
        Output{1,i} = zeros(Layer{i,2},Batchsize);
        Output{2,i} = zeros(Layer{i,2},Batchsize);
    else
        error('Layer must be called "Conv" (Convolution), "Pool" (Pooling) or "Full" (Fully Connected)!')
    end
end
if(regexp(Layer{end,1},'Input|Conv|Pool'))
    Weights{1,end} = std_init.*init_fct([size(Target,1) numel(Output{1,i+1})/size(Output{1,i+1},4)]);
    Weights{5,end} = @(x) reshape(x, [numel(x)/size(x,4) size(x,4)]);
    Weights{6,end} = @(x,y) reshape(x, size(y));
else
    Weights{1,end} = std_init.*init_fct([size(Target,1) Layer{i,2}]);
    Weights{5,end} = @(x) x;
    Weights{6,end} = @(x,y) x;
end
Weights{2,i-1} = [1 1 1];
Biases{1,end} = std_init.*init_fct([size(Target,1) 1]);
Output{1,end} = zeros(size(Target,1),Batchsize);
Output{2,end} = zeros(size(Target,1),Batchsize);
N_Layers = [N_Layers,'Target'];
Ntw{i+1,1} = sprintf('Target@(%i)',size(Target,1));
features = Weights{2,1}(3);


% Algorithm
Perform_meas = nan(nEpochs,2);
if(verbose > 1)
    wb_hdl = waitbar(0,sprintf('%i of %i batches finished!',0,floor(size(Input,3)/Batchsize)),'name','Epochs finished');
    if(verbose > 2)
        figure('Pos',[0 40 560 420]);
        errplot_hdl = axes(); xlabel('No. Epochs'); ylabel(ylab_plot); hold on
    end
end
fprintf('\n Start Training the Network!')
fprintf(['\n Network:  ',strjoin(Ntw(:)'),'\n\n'])
for t = 1:nEpochs
    
    shuff_vec = randperm(size(Input,3));
    
    Input = Input(:,:,shuff_vec);
    Target = Target(:,shuff_vec);
    
    % Batch Processing
    for b = 1:floor(size(Input,3)/Batchsize)
               
        Output{1,1} = permute(repmat(Input(:,:,(1:Batchsize)+Batchsize*(b-1)),[1 1 1 features]),[1 2 4 3]);
        Tar = Target(:,(1:Batchsize)+Batchsize*(b-1));
        dropout = cell(1,length(N_Layers)-1);
        
        % Forward
        for lyr = 1:length(N_Layers)-1
            if(strcmp(N_Layers{lyr},{'Input','Conv','Pool'}) && strcmp(N_Layers{lyr+1},{'Conv','Pool'}))
                if(strcmp(N_Layers{lyr},{'Input','Conv','Pool'}) && strcmp(N_Layers{lyr+1},'Conv'))
                    for p = 1:Batchsize
                        for f = 1:features
                            Output{2,lyr+1}(:,:,f,p) = convolve2(Output{1,lyr}(:,:,f,p), rot90(Weights{1,lyr}(:,:,f),2),'valid');
                        end
                    end
                else
                    help = reshape(Output{1,lyr},[Weights{3,lyr}(1) (1 + (size(Output{1,lyr},1)-Weights{3,lyr}(1))/Weights{3,lyr}(1)) Weights{3,lyr}(2) (1 + (size(Output{1,lyr},2)-Weights{3,lyr}(2))/Weights{3,lyr}(2)) features Batchsize]);
                    Output{2,lyr+1} = squeeze(max(max(help,[],1),[],3));
                    Weights{1,lyr} = reshape(logical(help.*bsxfun(@eq,help,max(max(help,[],1),[],3))),size(Output{1,lyr}));
                end
            else
                dropout{1,lyr} = rand(size(Weights{1,lyr},1),1) > drop;
                Output{2,lyr+1} = bsxfun(@times, bsxfun(@plus, Weights{1,lyr}*Weights{5,lyr}(Output{1,lyr}), Biases{lyr}), dropout{1,lyr})./drop;
            end
            Output{1,lyr+1} = Neuron_fct{2,lyr}(Output{2,lyr+1});
        end
        if(softmax)
            Output{1,end} = bsxfun(@rdivide, exp(Output{2,end}), sum(exp(Output{2,end}),1));
        end
        dropout{1,find(cellfun(@isempty, dropout),1,'last')} = drop;
        
        % Backward
        backw_layer_ind = 1:length(N_Layers);      
        backw_layer_ind([find(~cellfun(@isempty, regexp(N_Layers,'Pool')))-1 end]) = [];
        for lyr = backw_layer_ind
            delta = bsxfun(@times, Cost_fct{2,1}(Output{1,end}, Tar, Neuron_fct{3,end}(Output{2,end})), dropout{1,end})./drop;
            for bw_lyr = size(Weights,2):-1:lyr+1
                if(strcmp(N_Layers{bw_lyr},{'Input','Conv','Pool'}) && strcmp(N_Layers{bw_lyr+1},{'Conv','Pool'}))
                    help = zeros(size(Output{1,bw_lyr}));
                    for p = 1:Batchsize
                        for f = 1:features
                            w = Weights{9,bw_lyr}(Weights{1,bw_lyr},f,p);
                            help(:,:,f,p) = Weights{7,bw_lyr}(Weights{6,bw_lyr}(delta(:,:,f,p), ones(Weights{3,bw_lyr})), w);
                        end
                    end
                    delta = Weights{8,bw_lyr}(help, Neuron_fct{3,bw_lyr}(Output{2,bw_lyr}));
                else
                    delta = bsxfun(@times, Weights{6,bw_lyr}(Weights{1,bw_lyr}'*delta, Output{2,bw_lyr}) .* Neuron_fct{3,bw_lyr}(Output{2,bw_lyr}), dropout{1,bw_lyr-1})./drop;
                end
            end

            if(regexp(N_Layers{lyr},{'Input','Conv','Pool'}) && regexp(N_Layers{lyr+1},{'Conv','Pool'}))
                help = zeros(size(Weights{1,lyr}));
                for p = 1:Batchsize
                    for f = 1:Weights{2,lyr}(3)
                        help(:,:,f,p) = convolve2(Output{1,lyr}(:,:,f,p), rot90(delta(:,:,f,p),2), 'valid');
                    end
                end
                Weights{1,lyr} = bsxfun(@minus, Weights{lyr}, (lrate(t))*mean(help,4));
                Biases{1,lyr} = bsxfun(@minus, Biases{lyr}, (lrate(t))*mean(sum(sum(help,1),2),4));
            else
                Weights{1,lyr} = bsxfun(@minus, Weights{1,lyr}, (lrate(t)/Batchsize)*delta*Weights{5,lyr}(Output{1,lyr})');
                Biases{1,lyr} = bsxfun(@minus, Biases{1,lyr}, (lrate(t)/Batchsize)*sum(delta,2));
            end

        end
        
        if(verbose > 0 && mod(b,100) == 0)
            [~,pick] = max(Output{1,end},[],1);
            real = mod(find(Tar),10) + 10*(mod(find(Tar),10) == 0);
            fprintf('\n Epoch: %i  ---  %i Batches done  ---  Cost: %0.5f  ---  Acc: %0.3f',t,b,mean(Cost_fct{1,1}(Output{1,end}, Tar)),sum((pick'-real)==0)/Batchsize)
        end
        if(verbose > 1)
            waitbar((t-1)/nEpochs,wb_hdl,sprintf('%i of %i epochs finished! (%i / %i batches)',t-1,nEpochs,b,floor(size(Input,3)/Batchsize)),'name','Epochs finished');
        end
        
    end
    
    Output_test = cell(2,length(N_Layers)); % First row is sigma(z), second is z;
    Output_test{1,1} = permute(repmat(Input_test,[1 1 1 features]),[1 2 4 3]);
    % Forward
    for lyr = 1:length(N_Layers)-1
        if(strcmp(N_Layers{lyr},{'Input','Conv','Pool'}) && strcmp(N_Layers{lyr+1},{'Conv','Pool'}))
            if(strcmp(N_Layers{lyr},{'Input','Conv','Pool'}) && strcmp(N_Layers{lyr+1},'Conv'))
                for p = 1:size(Output_test{1,1},4)
                    for f = 1:features
                        Output_test{2,lyr+1}(:,:,f,p) = convolve2(Output_test{1,lyr}(:,:,f,p), rot90(Weights{1,lyr}(:,:,f),2),'valid');
                    end
                end
            else
                help = reshape(Output_test{1,lyr},[Weights{3,lyr}(1) (1 + (size(Output_test{1,lyr},1)-Weights{3,lyr}(1))/Weights{3,lyr}(1)) Weights{3,lyr}(2) (1 + (size(Output_test{1,lyr},2)-Weights{3,lyr}(2))/Weights{3,lyr}(2)) features size(Output_test{1,1},4)]);
                Output_test{2,lyr+1} = squeeze(max(max(help,[],1),[],3));
                Weights{1,lyr} = reshape(logical(help.*bsxfun(@eq,help,max(max(help,[],1),[],3))),size(Output_test{1,lyr}));
            end
        else
            Output_test{2,lyr+1} = bsxfun(@plus, Weights{1,lyr}*Weights{5,lyr}(Output_test{1,lyr}), Biases{lyr});
        end
        Output_test{1,lyr+1} = Neuron_fct{2,lyr}(Output_test{2,lyr+1});
    end
    if(softmax)
        Output_test{1,end} = bsxfun(@rdivide, exp(Output_test{2,end}), sum(exp(Output_test{2,end}),1));
    end
    
    [~,id] = max(Output_test{1,end});
    
    Perform_meas(t,1) = sum((id' - (mod(find(Target_test),10) + 10*(mod(find(Target_test),10) == 0))) == 0)/size(Target_test,2);
    Perform_meas(t,2) = mean(Cost_fct{1,1}(Output_test{1,end}, Target_test));
        
    if(verbose > 0)
        fprintf('\n Epoch No %i :  %i / %i correctly identified!',t,sum((id' - (mod(find(Target_test),10) + 10*(mod(find(Target_test),10) == 0))) == 0),size(Target_test,2))
        if(verbose > 2)
            if(strcmp(PerfPlot,'acc'))
                plot(errplot_hdl, Perform_meas(:,1), 'r', 'LineWidth', 2)
            else
                plot(errplot_hdl, Perform_meas(:,2), 'r', 'LineWidth', 2)
            end
            set(errplot_hdl,'xlim',[1 nEpochs])
        end
    end
    
end
fprintf('\n\n Training finished! \n\n')

end


