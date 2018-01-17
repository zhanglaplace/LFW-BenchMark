function align_KDEF()
    folder = 'E:\datasets\KDEF';
    
    %% 获取文件list
    image_list = get_image_list_in_folder(folder);
    save image_list.mat image_list;
    % load image_list.mat
    target_folder = 'E:\datasets\KDEF-112X96';
    if exist(target_folder, 'dir')==0
        mkdir(target_folder);
    end;

    pdollar_toolbox_path='E:\programs\ZF\sphereface\tools\toolbox';
    addpath(genpath(pdollar_toolbox_path));

    MTCNN_path = 'E:/programs/ZF/sphereface/tools/MTCNN_face_detection_alignment/code/codes/MTCNNv1';
    caffe_model_path=[MTCNN_path , '/model'];
    addpath(genpath(MTCNN_path));

    coord5points = [30.2946, 65.5318, 48.0252, 33.5493, 62.7299; ...
                    51.6963, 51.5014, 71.7366, 92.3655, 92.2041];
    imgSize = [112, 96];
    align_method = 'yandong';% wuxiang or yandong
            
    %caffe.set_mode_cpu();
    gpu_id=0;
    caffe.set_mode_gpu();	
    caffe.set_device(gpu_id);
    caffe.reset_all();

    %three steps's threshold
    %threshold=[0.6 0.7 0.7]
    threshold=[0.6 0.7 0.7];
    minsize = 20;%80;

    %scale factor
    factor=0.85;%0.709;

    %load caffe models
    PNet = caffe.Net('E:/programs/ZF/sphereface/tools/MTCNN_face_detection_alignment/code/codes/MTCNNv1/model/det1.prototxt','E:/programs/ZF/sphereface/tools/MTCNN_face_detection_alignment/code/codes/MTCNNv1/model/det1.caffemodel', 'test');
    RNet = caffe.Net('E:/programs/ZF/sphereface/tools/MTCNN_face_detection_alignment/code/codes/MTCNNv1/model/det2.prototxt', ...
                     'E:/programs/ZF/sphereface/tools/MTCNN_face_detection_alignment/code/codes/MTCNNv1/model/det2.caffemodel', 'test');
    ONet = caffe.Net('E:/programs/ZF/sphereface/tools/MTCNN_face_detection_alignment/code/codes/MTCNNv1/model/det3.prototxt', ...
                     'E:/programs/ZF/sphereface/tools/MTCNN_face_detection_alignment/code/codes/MTCNNv1/model/det3.caffemodel', 'test');
    faces=cell(0);	

    for image_id = 1:length(image_list)
        img = imread(image_list{image_id});
        if size(img, 3) < 3
           img(:,:,2) = img(:,:,1);
           img(:,:,3) = img(:,:,1);
        end
        [file_folder, file_name, file_ext] = fileparts(image_list{image_id});
        target_filename = strrep(image_list{image_id},folder, target_folder);
        assert(strcmp(target_filename, image_list{image_id})==0);
        [file_folder, file_name, file_ext] = fileparts(target_filename);
        if exist(file_folder,'dir')==0
            mkdir(file_folder);
        end;
        if exist(target_filename,'file') ~= 0
            continue;
        end
        disp([num2str(image_id) '/' num2str(length(image_list)) ' ' target_filename]);
        [boundingboxes, points] = detect_face(img, minsize, PNet, RNet, ONet, threshold, false, factor);

        if isempty(boundingboxes)
            continue;
        end;
        default_face = 1;
        if size(boundingboxes,1) > 1
            for bb=2:size(boundingboxes,1)
                if abs((boundingboxes(bb,1) + boundingboxes(bb,3))/2 - size(img,2) / 2) + abs((boundingboxes(bb,2) + boundingboxes(bb,4))/2 - size(img,1) / 2) < ...
                        abs((boundingboxes(default_face,1) + boundingboxes(default_face,3))/2 - size(img,2) / 2) + abs((boundingboxes(default_face,2) + boundingboxes(default_face,4))/2 - size(img,1) / 2)
                    default_face = bb;
                end;
            end;
        end;
        facial5points = double(reshape(points(:,default_face),[5 2])');
        if strcmp(align_method, 'wuxiang') > 0
            [res, eyec2, cropImg, resize_scale] = align_face_WX(img,facial5points',144,48,48);
            cropImg = uint8(cropImg);
        else
            Tfm =  cp2tform(facial5points', coord5points', 'similarity');
            cropImg = imtransform(img, Tfm, 'XData', [1 imgSize(2)],...
                                          'YData', [1 imgSize(1)], 'Size', imgSize);
        end;
        imwrite(cropImg, target_filename);

    end

end
function [ image_list ] = get_image_list_in_folder( folder )
    root_list = dir(folder);
    root_list = root_list(3:end);
    image_list = {};
    for i=1:length(root_list)
        if root_list(i).isdir
            sub_list = get_image_list_in_folder(fullfile(folder,root_list(i).name));
            image_list = [image_list;sub_list];
        else
            [~, ~, c] = fileparts(root_list(i).name);
            if strcmp(c,'.png') == 0 && strcmp(c,'.jpg') == 0 && strcmp(c,'.bmp') == 0 && strcmp(c,'.jpeg') == 0 && strcmp(c,'.tiff') == 0 ...
                && strcmp(c,'.PNG') == 0 && strcmp(c,'.JPG') == 0 && strcmp(c,'.BMP') == 0 && strcmp(c,'.JPEG') == 0 && strcmp(c,'.TIFF') == 0
                continue;
            end;
            image_list = [image_list;fullfile(folder,root_list(i).name)];
        end;
    end;
end