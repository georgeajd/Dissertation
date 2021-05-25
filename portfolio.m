%**************************************************************************
%
%https://uk.mathworks.com/matlabcentral/fileexchange/18458-hist_stock_data-...
%start_date-end_date-varargin
stockticks = importdata('StockTickers.txt');
SZSTCKS = 10;  
dateBeg = '01012018'; 
dateEnd = '06102020'; 
DATLEN = 695;    
marketData = zeros(SZSTCKS,DATLEN); 
tickprices = zeros(DATLEN,1); 
STKS = 0;
for k = 1:SZSTCKS
    disp(stockticks{k});
    data = hist_stock_data(dateBeg,dateEnd,stockticks{k});
    if (isempty(data))
        disp('?');
    else
        if (size(data.Close,1)<695)
            disp('!');
        else
            tickprices = flipud(data.Close); %tprices = tickprices;
            for t = DATLEN:-1:1 %for t=1:DATLEN
                marketData(k,t) = tickprices(t);
            end
            STKS = STKS+1;
        end
    end
end

%save stocks to local drive
fid = fopen('C:\Users\George\Documents\Dissertation\Code\alSPnov.dat','w');
for t = 1:DATLEN
    fprintf(fid,'\n');
    for k = 1:STKS
        fprintf(fid,'%4.2f \t', marketData(k,t));  
    end
end
fclose(fid);

load alSPnov.dat;
[DATLEN,SZSTCKS] = size(alSPnov);
COL = SZSTCKS;  
adata = zeros(DATLEN,SZSTCKS);
for t = 1:DATLEN
    for k = 1:SZSTCKS
        adata(t,k) = alSPnov(t,k);
    end
end
%..........................................................................
% Invert the sequence in time
invadata = zeros(DATLEN,SZSTCKS);
for t = 1:DATLEN
    for k = 1:SZSTCKS
        invadata(DATLEN-t+1,k) = adata(t,k);
    end
end
%--------------------------------------------------------------------------
% Calculating Returns
idata = zeros(DATLEN,COL);
for t = 2:DATLEN
    for j = 1:COL
        idata(t-1,j) = invadata(t,j)./invadata(t-1,j)-1;
    end  
end
idata(DATLEN,:) = []; %removing last row of empty values caused by returns calculations
%--------------------------------------------------------------------------
% Applying centered moving average to series
[DATLEN,SZSTCKS] = size(idata);
for t = 2:DATLEN-1
    for k = 1:SZSTCKS
        idata(t,k) = ((0.8*(idata(t-1,k)))+(1.4*(idata(t,k)))+(0.8*(idata(t+1,k))))/3;
    end
end

idata = idata'*100; 
PREDICTIONS = [];
for k = 1:SZSTCKS
    Ss=idata(k,:);
    idim=5; % input dimension
    odim=length(Ss)-idim; % output dimension
    for i=1:odim
       y(i)=Ss(i+idim);
       for j=1:idim
           x(i,j) = Ss(i-j+idim); 
       end
    end
    Patterns = x(odim-400:odim,:)'; Testing = x(400:odim,:)';   
    Desired = y(odim-400:odim); TestingDesired = y(400:odim);
    TESTSIZE = odim-400+1; PATSIZE = 401;
    
    [NINPUTS,NPATS] = size(Patterns); [NOUTPUTS,NP] = size(Desired);
    NHIDDENS = 5; LearnRate = 0.01; deltaW1 = 0; deltaW2 = 0; deltaW3 = 0;
    Weights1 = 0.5*(rand(NHIDDENS,NINPUTS)-0.5);
    Weights2 = 0.5*(rand(1,NHIDDENS)-0.5); 
    
    %TRAINING
    for epoch = 1:200
        for i = 1:length(Patterns)
          % Forward propagation
          NetIn1 = Patterns(:,i);
          Hidden = 1.0 ./( 1.0 + exp( -(NetIn1' * Weights1') ));
          NetIn2 = Hidden*Weights2';
          Out = tanh(NetIn2);
          PREDICTIONS(k,i) = Out;
          
          % Backward propagation of errors
          Delta = Out-Desired(i);
          bperr = Delta*(1-Out^2);
          deltaW2 = Weights2-(LearnRate*(bperr*Hidden)); 
          HiddenBeta = (bperr*Weights2).*Hidden.*(1-Hidden);
          deltaW1 = Weights1-(LearnRate*HiddenBeta'*NetIn1');
           
          % Update the weights:
          Weights2 = deltaW2;
          Weights1 = deltaW1;

        end
    end

    %TESTING
    for i = 1:length(Testing),
        % Forward propagation
        NetIn1 = Testing(:,i);
        Hidden = 1.0 ./( 1.0 + exp( -(NetIn1' * Weights1') ));
        NetIn2 = Hidden * Weights2';
        Out = NetIn2;
        PREDICTIONS(k,i+PATSIZE) = Out;
    end
   
end

%RETURNS = 1 step ahead forecasts at each timestep from the network
m = SZSTCKS; % number of stocks
adjweightsmatrix = zeros(m,TESTSIZE);
for i = PATSIZE+1:PATSIZE+TESTSIZE,
    range = 5; 
    RETURNS = PREDICTIONS(:,i-range:i);
    
    cmus = mean(RETURNS,2); 
    Xf = RETURNS/sqrt(range); 
    Shat = zeros(m,m);  
    for ic = 1:m
        Shat(ic,ic) = (Xf(ic,:)*Xf(ic,:)');
        for jc = ic+1:m
            Shat(ic,jc) = (Xf(ic,:)*Xf(jc,:)');
            Shat(jc,ic) = Shat(ic,jc);
        end
    end
    Sigi = Shat\eye(m); % Sigi = pinv(Shat); % Sigi = inv(Shat(:,:));
    
    gamh = 1.0; 
    onetranS = ones(m,1)'*Sigi; % utility portfolio
    lambt = (gamh-onetranS'*cmus')/(onetranS*ones(m,1));
    aweight = (1/gamh)*Sigi*(cmus+lambt*ones(m,1));% NN weighted portfolio
    %aweight = 1/m;% for equally weighted portfolio
    
    %{
    %Adjusting the weights to sum to 1
    adjweights = [];
    for j = 1:m
        adjweights(j) = aweight(j)/sum(aweight);
    end
    adjweightsmatrix(:,i-PATSIZE) = adjweights;
    %}
    
    adjweightsmatrix(:,i-PATSIZE) = aweight;
end

testreturns = PREDICTIONS(:,odim+3-TESTSIZE:end);%returns from test set predictions
testreturnssum = cumsum(testreturns,2);
testreturnssum = sum(testreturnssum);

adjustedprofit = adjweightsmatrix .* testreturns;
adjustedprofitsum = cumsum(adjustedprofit,2);
adjustedprofitsum = sum(adjustedprofitsum);



stockplot = [adjustedprofitsum;testreturnssum];
plot(stockplot');
legend('Adjusted Returns','Original Returns');



