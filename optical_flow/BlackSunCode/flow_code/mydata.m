METHOD = 'hs';

outdir = '/Users/michaellavender/Documents/BUSPC/optical flow/BA/hs';
if ~exist(outdir, 'dir'); mkdir(outdir); end

frame_folder = '/Users/michaellavender/Documents/BUSPC/optical flow/frames';

cropRows = 1:490;
frame_i_start = 3195;
frame_i_end = 3313;
frame_step = 2;
frame_indicies = frame_i_start:frame_step:frame_i_end;

pairs = [frame_indicies(1:end-1).', frame_indicies(2:end).']; %pairwise of frames

gain = 2; %gain for vector scaling

fig = figure(Visible="off");
ax = axes(Parent=fig);

N = size(pairs, 1);
t0 = tic;
wb = waitbar(0, 'Starting…', 'Name', 'Optical flow', ...
             'CreateCancelBtn', 'setappdata(gcbf, ''cancel'', 1)');
setappdata(wb, 'cancel', 0);

for i = 1:N
    if getappdata(wb, 'cancel'); break; end

    f1_i = pairs(i, 1); %frame index of first frame
    f2_i = pairs(i, 2);

    fn1 = fullfile(frame_folder, sprintf('frame%d.png', f1_i)); % ex framefolder/frame1
    fn2 = fullfile(frame_folder, sprintf('frame%d.png', f2_i));

    im1 = double(imread(fn1));
    im2 = double(imread(fn2));
    im1 = im1(cropRows, :, :); % crop 
    im2 = im2(cropRows, :, :);

    uv = estimate_flow_interface(im1, im2, METHOD);

    step = 8; % arrow spacing
    [X, Y] = meshgrid(1:step:size(uv, 2), 1:step:size(uv,1));
    u = uv(1:step:end, 1:step:end, 1);
    v = uv(1:step:end, 1:step:end, 2);

    outfn = fullfile(outdir, sprintf('frame%d-%d.png', f1_i, f2_i));

    cla(ax);
    imshow(uint8(im1), Parent=ax); hold(ax, "on");
    quiver(ax, X, Y, u*gain, v*gain, 0, 'y', LineWidth=1.2);
    hold(ax, "off");
    exportgraphics(ax, outfn, 'Resolution', 150);
    save(fullfile(outdir, sprintf('flow%d-%d.mat', f1_i, f2_i)), 'uv');

    elapsed = toc(t0);
    eta     = elapsed/i * (N - i);
    waitbar(i/N, wb, sprintf('%d / %d   —   %s remaining', ...
        i, N, string(duration(0, 0, round(eta)))));

end

close(fig);    
close(wb);







% % save
% exportgraphics(gca, 'flow_overlay.png', 'Resolution', 300);

% % save 

% % declutter
% mag = sqrt(u.^2 + v.^2);
% mask = mag > 0.3;             % pixels/frame threshold
% quiver(X(mask), Y(mask), u(mask), v(mask), 0, 'y', 'LineWidth', 1.2);


