function  view_subplot( datas ,samples,value_title,clusterNums,figName)
    close all;
    figure('rend','painters','pos', [1200 900 1500 600])
    count = size(value_title,2);
    for i_sub=1:count
        subplot(2,floor((count+1)/2),i_sub);   
%         title(value_title);
        hold on;
        data_sample = samples{i_sub};
        mean_sample = mean(data_sample);
        var_sample = std(data_sample);
        %boxplot(data_sample, clusterNums);  
        errorbar(clusterNums,mean_sample,var_sample,'-s','MarkerSize',6,...
          'MarkerEdgeColor','b','MarkerFaceColor','b')
        
        data_sub = {datas{i_sub,:,1}}';
        for i=1:size(data_sub,1)
              data = data_sub{i,1};
              x = data(:,1);
%               x= [1,2,3,4,5,6,7];
              y = data(:,2);
              
              plot(x,y,'-*');
              %yticks([0 1e-5 1e-3 1e-2 1e-1])
              axis([0 40 1e-3 5e-1])
              %ylim([0,8E-1])
              %semilogx(x,y)
              

              
        end
        labels = {datas{i_sub,:,2}};
        labels = cellstr(labels);
        labels = {'GT' labels{:}};
        h=legend(labels,'Location','southeastoutside');
        title(value_title{i_sub});
        set(h,'Fontsize',10);
        %set(gca,'Fontsize',10) ;
        set(gca, 'YScale', 'log','Fontsize',10)

        
        grid on      
    end
    print([figName, '.eps'],'-depsc','-r0');
end

