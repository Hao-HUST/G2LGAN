function view_singleplot( datas,samples,value_title,clusterNums, figName, shapes)
    figure;
    title(value_title);

    hold on;
    mean_sample = mean(samples);
    var_sample = std(samples);
    errorbar(clusterNums,mean_sample,var_sample,'-s','MarkerSize',6,...
      'MarkerEdgeColor','b','MarkerFaceColor','b')
    
    for i=1:size(datas,1)
      data = datas{i,1};
      x = data(:,1);
%       x= [1,2];
      y = data(:,2);
      axis([0 40 1e-2 1])
      %plot(x,y,shapes{i});
      plot(x,y,'-*');
      hold on;
    end

    labels = {datas{:,2}};
    labels = cellstr(labels);
    labels = {'GT' labels{:}};
    h=legend(labels,'Location','southeastoutside');
    title(value_title);
    set(h,'Fontsize',10);
    set(gca,'Fontsize',10) ;
    set(gca, 'YScale', 'log','Fontsize',10)
    grid on 
    
    print([figName, '.eps'],'-depsc','-r0');
end
