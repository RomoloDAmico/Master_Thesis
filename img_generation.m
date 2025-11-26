function img_generation(x, output_folder)
% ==========================================================
% GENERATORE DI FRAMES PER BRACCIO DOPPIO LINK (PARAMETRI INTERNI)
% ==========================================================

    % Parametri del robot — fissi
    l1 = 1;
    l2 = 0.7;

    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end

    figure(1);
    clf;
    axis equal;
    xlim([-(l1 + l2), (l1 + l2)]);
    ylim([-(l1 + l2), (l1 + l2)]);
    set(gca, 'XColor', 'none', 'YColor', 'none');
    set(gcf, 'Color', 'w');
    set(gca, 'Color', 'none');
    hold on;

    N = size(x, 2);

    for i = 1:N
        cla;

        q1 = x(1,i);
        q2 = x(2,i);

        % coordinate giunti
        P1x = l1 * cos(q1);
        P1y = l1 * sin(q1);
        P2x = P1x + l2 * cos(q1 + q2);
        P2y = P1y + l2 * sin(q1 + q2);

        % disegna i link
        line([0 P1x], [0 P1y], 'LineWidth', 6, 'Color', 'k');
        line([P1x P2x], [P1y P2y], 'LineWidth', 6, 'Color', 'k');

        % disegna i giunti
        plot(0, 0, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 8);
        plot(P1x, P1y, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 8);
        plot(P2x, P2y, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 8);

        drawnow;

        set(gcf,'Color','w');
        set(gca,'Color','none');

        % salva frame
        filename = sprintf('%s/frame_%04d.png', output_folder, i);
        saveas(gcf, filename);
    end
end
