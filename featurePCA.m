% Principle component analysis and visualization
% I made this its own function so it can be run separately without having
% to run the whole pipeline
function [] = featurePCA(data)
    tic
    [~,scrs,~,~,pexp] = pca(double(data.features));
    data_reduced = scrs(:,1:3);
    pca_runtime = toc;
    fprintf("Performed PCA in: %.8f seconds\n",pca_runtime);
    
    pareto(pexp);
    title('Result of PCA, first 10 components');
    
    colors = 'rb';
    markers = 'xo';
    figure;
    for cls=0:1
        idx = (data.class == cls);
        x = data_reduced(idx,:);
        plot3(x(:,1),x(:,2),x(:,3), [colors(cls+1) markers(cls+1)]);
        hold on
        xlabel('Component 1');
        ylabel('Component 2');
        zlabel('Component 3');
        title('3 Most dominant components');
    end
end