frame_folder = '/Users/michaellavender/Documents/BUSPC/optical flow/frames';
mat_folder = '/Users/michaellavender/Documents/BUSPC/optical flow/BA/hs/uv';
outdir = '/Users/michaellavender/Documents/BUSPC/optical flow/BA/hs/output-frames-filtered';
if ~exist(outdir, 'dir'); mkdir(outdir); end


cropRows = 1:490;
frame_i_start = 3195;
frame_i_end = 3313;
frame_step = 2;
frame_indicies = frame_i_start:frame_step:frame_i_end;
pairs = [frame_indicies(1:end-1).', frame_indicies(2:end).']; %pairwise of frames
N = size(pairs, 1);          


sigma = 1.5;      % Gaussian sigma in pixels (display only)
gain  = 3;        % arrow length scaling
step  = 20;        % arrow spacing

fig = figure(Visible="off");
ax = axes(Parent=fig);

for i = 1:N
    f1_i = pairs(i, 1); %frame index of first frame
    f2_i = pairs(i, 2);

    matfn = fullfile(mat_folder, sprintf('flow%d-%d.mat', f1_i, f2_i));
    if ~isfile(matfn)
        warning('missing %s — skipping', matfn);
        continue
    end

    S  = load(matfn, 'uv');
    uv = S.uv;



    fn1 = fullfile(frame_folder, sprintf('frame%d.png', f1_i)); % ex framefolder/frame1

    im1 = double(imread(fullfile(frame_folder, sprintf('frame%d.png', f1_i))));
    im1 = im1(cropRows, :, :);
    if size(im1,3) > 1
        im1 = im1(:,:,1);
    end
    im1s = imgaussfilt(im1, sigma);     % <-- the Gaussian

    [X, Y] = meshgrid(1:step:size(uv, 2), 1:step:size(uv,1));
    u = uv(1:step:end, 1:step:end, 1);
    v = uv(1:step:end, 1:step:end, 2);

    cla(ax);
    imshow(im1s, [], 'Parent', ax); hold(ax, "on");
    quiver(ax, X, Y, u*gain, v*gain, 0, 'y', LineWidth=1.2);
    hold(ax, "off");


    outfn = fullfile(outdir, sprintf('frame%d-%d.png', f1_i, f2_i));
    exportgraphics(ax, outfn, 'Resolution', 150);

end

close(fig);