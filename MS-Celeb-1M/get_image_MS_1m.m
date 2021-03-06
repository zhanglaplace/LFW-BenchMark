function get_image_MS_1m()
    root_folder = 'D:\others\FacialExpressionImage\MegaFace\FlickrFinal2';
    
    %% 获取图片
%     if ~exist('image_list','var')
%         list_file = 'D:\others\FacialExpressionImage\MegaFace\devkit\templatelists\megaface_features_list.json_1000000_1';
%         json_string = fileread(list_file);
%         image_list = regexp(json_string(8:end), '"(.*?)"','tokens');
%         for i=1:length(image_list)
%             image_list{i} = [root_folder '/' image_list{i}{1}];
%         end;
%         save image_list.mat image_list
%     end;
    load image_list.mat
    
    %% 输出文件
    
    noface_file = 'D:/others/FacialExpressionImage/MegaFace/noface_detect.txt';
    noface_fid = fopen(noface_file,'w');
    
    %% 保存目录
    target_folder = 'D:/others/FacialExpressionImage/MegaFace/FlickrFinal2-112X96';
    if exist(target_folder, 'dir')==0
        mkdir(target_folder);
    end;
    
    pdollar_toolbox_path='D:/programs/sphereface/tools/toolbox';
    addpath(genpath(pdollar_toolbox_path));

    MTCNN_path = 'D:/programs/sphereface/tools/MTCNN_face_detection_alignment/code/codes/MTCNNv1';
    caffe_model_path = [MTCNN_path , '/model'];
    addpath(genpath(MTCNN_path));

    %% align Setting
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
    threshold=[0.6 0.7 0.9];
    minsize = 20;%80;

    %scale factor
    factor=0.85;%0.709;

    %% load caffe models
    PNet = caffe.Net('D:/programs/sphereface/tools/MTCNN_face_detection_alignment/code/codes/MTCNNv1/model/det1.prototxt','D:/programs/sphereface/tools/MTCNN_face_detection_alignment/code/codes/MTCNNv1/model/det1.caffemodel', 'test');
    RNet = caffe.Net('D:/programs/sphereface/tools/MTCNN_face_detection_alignment/code/codes/MTCNNv1/model/det2.prototxt', ...
                     'D:/programs/sphereface/tools/MTCNN_face_detection_alignment/code/codes/MTCNNv1/model/det2.caffemodel', 'test');
    ONet = caffe.Net('D:/programs/sphereface/tools/MTCNN_face_detection_alignment/code/codes/MTCNNv1/model/det3.prototxt', ...
                     'D:/programs/sphereface/tools/MTCNN_face_detection_alignment/code/codes/MTCNNv1/model/det3.caffemodel', 'test');
    faces = cell(0);	

    
    
    
    %% 循环处理每张图片
    for image_id = 1:length(image_list)
        img = imread(image_list{image_id});
        flag_resize = false;
        if size(img,1) > 1600 
            img = imresize(img,[size(img,1)/4,size(img,2)/4]);
            flag_resize = true;
        end
        if size(img,2) > 1600
            img = imresize(img,[size(img,1)/4,size(img,2)/4]);
             flag_resize = true;
        end
        if size(img, 3) < 3
           img(:,:,2) = img(:,:,1);
           img(:,:,3) = img(:,:,1);
        end
        
        %% 目标文件目录target_filename
        [file_folder, file_name, file_ext] = fileparts(image_list{image_id});
        target_filename = strrep(image_list{image_id},root_folder, target_folder);
        assert(strcmp(target_filename, image_list{image_id})==0);
        [file_folder, file_name, file_ext] = fileparts(target_filename);
        if exist(file_folder,'dir')==0
            mkdir(file_folder);
        end;
        
        %% 目标文件存在，则继续
        if exist(target_filename,'file') ~= 0
            continue;
        end
       
        disp([num2str(image_id) '/' num2str(length(image_list)) ' ' target_filename]);
        
        [boundingboxes, points] = detect_face(img, minsize, PNet, RNet, ONet, threshold, false, factor);
        
        
        %% 如果检测到人脸
        default_face = 1;
         %% 如果json文件存在，则通过mtcnn检测
        json_filename = [image_list{image_id},'.json'];
        if exist(json_filename,'file') ~= 0
            json = parse_json(fileread(json_filename));
            coarse_box = [json{1}.bounding_box.x json{1}.bounding_box.y json{1}.bounding_box.width json{1}.bounding_box.height];
            if flag_resize
                coarse_box = coarse_box/4;
            end
            if ~isempty(boundingboxes)
                if size(boundingboxes,1) > 1 %人脸数量大于1，选取与标签IOU最大的
                    for i = 2:size(boundingboxes,1)
                        if IoU(boundingboxes(i,1:4),coarse_box)> IoU(boundingboxes(default_face,1:4), coarse_box)
                              default_face = i;
                        end
                    end
                end
                
               %% 此时仅包含一个人脸
               if IoU(boundingboxes(default_face,:), coarse_box) > 0.25
                   force_detect = false;
               else
                   force_detect = true;
               end
            %% 如果根本没检测到人脸   
            else
                force_detect = true;
            end
            
            %%% mtcnn 检测的不好，那就用json文件的内容来对齐
            if 	force_detect
                roi = coarse_box;
                roi(3) = coarse_box(4) / imgSize(1) * imgSize(2);
                roi(1) = coarse_box(1) + (coarse_box(3) - roi(3)) / 2;
                bounding_box = floor(roi);
                default_face = 1;
                cropImg = imcrop(img,roi);
                cropImg = imresize(cropImg,imgSize);
            else % 检测的与json文件的iou大于0.25检测的来做对齐
                 facial5points = double(reshape(points(:,default_face),[5 2])');
                 if strcmp(align_method, 'wuxiang') > 0
                    [res, eyec2, cropImg, resize_scale] = align_face_WX(img,facial5points',144,48,48);
                    cropImg = uint8(cropImg);
                 else
                    Tfm =  cp2tform(facial5points', coord5points', 'similarity');
                    cropImg = imtransform(img, Tfm, 'XData', [1 imgSize(2)],...
                                          'YData', [1 imgSize(1)], 'Size', imgSize);
                 end
            end
            imwrite(cropImg,target_filename);
        %% json文件不存在  ,只能依靠mtcnn了  
        else
            % mtcnn也gg啦
            if isempty(boundingboxes)
                try
                    fprintf(noface_fid,[ image_list{image_id}  'Not Detect file']); 
                catch
                    disp(image_list{image_id});
                end
                continue;
            end;
            if size(boundingboxes,1) > 1
                for bb = 2:size(boundingboxes,1)
                    if abs((boundingboxes(bb,1) + boundingboxes(bb,3))/2 - size(img,2) / 2) + abs((boundingboxes(bb,2) + boundingboxes(bb,4))/2 - size(img,1) / 2) < ...
                            abs((boundingboxes(default_face,1) + boundingboxes(default_face,3))/2 - size(img,2) / 2) + abs((boundingboxes(default_face,2) + boundingboxes(default_face,4))/2 - size(img,1) / 2)
                        default_face = bb;
                    end;
                end;
            end
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
    fclose(noface_fid);
end



function overlap_rate = IoU(bbox1, bbox2)
    intersect_bbox = [max(bbox1(1), bbox2(1)) max(bbox1(2), bbox2(2)) min(bbox1(1)+bbox1(3), bbox2(1)+bbox2(3)) min(bbox1(2)+bbox1(4), bbox2(2)+bbox2(4))];
    overlap = (intersect_bbox(3) - intersect_bbox(1)) * (intersect_bbox(4) - intersect_bbox(2));
    overlap_rate = overlap / (bbox1(3)*bbox1(4) + bbox2(3)*bbox2(4) - overlap);
end