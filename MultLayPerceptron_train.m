function [Weights, Biases, Perform_meas] = MultLayPerceptron_train(Input, Target, Layer, lrate, nEpochs, Batchsize, varargin)

%MultLayPerceptron_train Train a Multilayer Perceptron Deep Neural Network on some Input data
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% SYNTAX:
% [Weights, Biases, Perform_meas] = MultLayPerceptron_train(Input, Target, Layer, lrate, nEpochs, Batchsize, varargin)
%
% INPUTS:
%   Input   Two dimensional (MxN) matrix of input images. M is the number
%           of input neurons per image while N is the number of all images
%           used. (Example: 784x60000 training images)
%   Target  Nx1 vector with containing the true label of each input image
%           as an integer indicating the target neuron the images category
%           belongs to.
%   Layer   1xN_Layer integer vector where each entry represents a deep 
%           layer with an integer as the number of neurons in this layer.
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
    Neuron_fct = repmat(Neuron_fct,1,length(Layer)+1);
end

for lyr = 1:length(Neuron_fct)
    
    % Network structure (assuming full connectedness)
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
                init_fct = @(x,y) randn(x,y);
            elseif(strcmp(init{2},'uni'))
                init_fct = @(x,y) rand(x,y);
            else
                error('Second entry of init has to be either a string (options: "uni" for uniform distr, "gauss" for stand norm distr) or function_handle to an anonymous initialization function!')
            end
        else
            std_init = init{1};
            init_fct = @(x,y) randn(x,y);
        end
    else
        std_init = init;
        init_fct = @(x,y) randn(x,y);
    end
else
    error('init has to be a cell with one (std value) or two (std value, anonymous initialization function)!')
end


% Get Initial Weights, Biases and Outputs
Ntw = {sprintf('Input@(%i) --> ', size(Input,1))};
N_Layers = [size(Input,1) Layer size(Target,1)];
Weights = cell(1,length(N_Layers)-1);
Biases = cell(1,length(N_Layers)-1);
Output = cell(2,length(N_Layers)); % First row is sigma(z), second is z
for lyr = 1:length(N_Layers)-1
    Ntw{lyr+1,1} = sprintf('Full@(%i) --> ', N_Layers(lyr+1));
    Weights{lyr} = std_init.*init_fct(N_Layers(lyr+1), N_Layers(lyr));
    Biases{lyr} = std_init.*init_fct(N_Layers(lyr+1),1);
end
Ntw{lyr+1,1} = sprintf('Target@(%i)', N_Layers(lyr+1));

% Algorithm
Perform_meas = nan(nEpochs,2);
if(verbose > 1)
    wb_hdl = waitbar(0,sprintf('%i of %i batches finished!',0,floor(size(Input,2)/Batchsize)),'name','Epochs finished');
    if(verbose > 2)
        figure('Pos',[0 40 560 420]);
        errplot_hdl = axes(); xlabel('No. Epochs'); ylabel(ylab_plot); hold on
    end
end
fprintf('\n Start Training the Network!')
fprintf(['\n Network:  ',strjoin(Ntw(:)'),'\n\n'])
for t = 1:nEpochs
    
    shuff_vec = randperm(size(Input,2));
    
    Input = Input(:,shuff_vec);
    Target = Target(:,shuff_vec);
    
    % Batch Processing
    for b = 1:floor(size(Input,2)/Batchsize)
        
        Output{1,1} = Input(:,(1:Batchsize)+Batchsize*(b-1));
        Tar = Target(:,(1:Batchsize)+Batchsize*(b-1));
        dropout = cell(1,length(N_Layers)-1);
        
        % Forward
        for lyr = 1:length(N_Layers)-1
            dropout{1,lyr} = rand(size(Weights{lyr},1),1) > drop;
            Output{2,lyr+1} = bsxfun(@times, bsxfun(@plus, Weights{lyr}*Output{1,lyr}, Biases{lyr}), dropout{1,lyr})./drop;
            Output{1,lyr+1} = Neuron_fct{2,lyr}(Output{2,lyr+1});
        end
        if(softmax)
            Output{1,end} = bsxfun(@rdivide, exp(Output{2,end}), sum(exp(Output{2,end}),1));
        end
        
        % Backward        
        for lyr = 1:length(N_Layers)-1
            delta = bsxfun(@times, Cost_fct{2,1}(Output{1,end}, Tar, Neuron_fct{3,end}(Output{2,end})), dropout{1,end});
            for bw_lyr = size(Weights,2):-1:1+lyr
                delta = bsxfun(@times, Weights{bw_lyr}'*delta.*Neuron_fct{3,bw_lyr}(Output{2,bw_lyr}), dropout{1,bw_lyr-1});
            end
            Weights{lyr} = bsxfun(@minus, Weights{lyr}, (lrate(t)/Batchsize)*delta*Output{1,lyr}');
            Biases{lyr} = bsxfun(@minus, Biases{lyr}, (lrate(t)/Batchsize)*sum(delta,2));
            clear temp_prod delta
        end
        
        
        if(verbose > 0 && mod(b,100) == 0)
            [~,pick] = max(Output{1,end},[],1);
            real = mod(find(Tar),10) + 10*(mod(find(Tar),10) == 0);
            fprintf('\n Epoch: %i  ---  %i Batches done  ---  Cost: %0.5f  ---  Acc: %0.3f',t,b,mean(Cost_fct{1,1}(Output{1,end}, Tar)),sum((pick'-real)==0)/Batchsize)
        end
        if(verbose > 1)
            waitbar((t-1)/nEpochs,wb_hdl,sprintf('%i of %i epochs finished! (%i / %i batches)',t-1,nEpochs,b,floor(size(Input,2)/Batchsize)),'name','Epochs finished');
        end

    end
    
    Output_test = cell(2,length(N_Layers)); % First row is sigma(z), second is z;
    Output_test{1,1} = Input_test;
    % Forward
    for lyr = 1:length(N_Layers)-1
       Output_test{2,lyr+1} = bsxfun(@plus, Weights{lyr}*Output_test{1,lyr}, Biases{lyr});
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
