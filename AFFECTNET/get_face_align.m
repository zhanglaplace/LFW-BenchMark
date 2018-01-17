function get_face_align()
    root_folder = 'F:\datasets\face_expression\AffectNet\Manually_Annotated_Images';
    valid_csv = 'F:\datasets\face_expression\AffectNet\affectNet_validation.csv';
    target_folder = 'F:\datasets\face_expression\AffectNet\Manually_Annotated_Images-112X96-Validate';
    if exist(target_folder,'dir') == 0
           mkdir(root_folder);
    end
    
    coord5points = [30.2946, 65.5318, 48.0252, 33.5493, 62.7299; ...
                51.6963, 51.5014, 71.7366, 92.3655, 92.2041];
    imgSize = [112, 96];
    
<<<<<<< Updated upstream
    csv_fid = fopen(valid_csv,'r');
    total_num = 5500;%%414800;
=======
    csv_fid = fopen(train_csv,'r');
    total_num = 414800; % 5500 validation
>>>>>>> Stashed changes
    tmp_line = fgetl(csv_fid); %% 第一行不要
    image_id = 0;
    while ~feof(csv_fid)
       
        image_id = image_id+1;
        
        %% 解析每一行
         line = fgetl(csv_fid);
         C = strsplit(line,',');
         if length(C) ~= 9
             continue
         end
         
          
         %% 只需要7种表情
         if str2num(C{1,7}) >= 7
             continue;
         end
         
        %% 判断face或者landmarks是否合理
        face_x = str2num(C{1,2});
        face_y = str2num(C{1,3});
        face_w = str2num(C{1,4});
        face_h = str2num(C{1,5});
        
        %% landmark 
        landmarks = str2num(C{1,6});
        if size(landmarks,1) ~= 136
            continue;
        end
        
        left_eye_x = sum(landmarks(73:2:83,1))/6;
        left_eye_y = sum(landmarks(74:2:84,1))/6;
        right_eye_x = sum(landmarks(85:2:95,1))/6;
        right_eye_y = sum(landmarks(86:2:96,1))/6;
        nip_x = landmarks(58,1);
        nip_y = landmarks(59,1);
        left_mouth_x = landmarks(97,1);
        left_mouth_y = landmarks(98,1);
        right_mouth_x = landmarks(109,1);
        right_mouth_y = landmarks(110,1);
        facial5points = [left_eye_x,right_eye_x,nip_x,left_mouth_x,right_mouth_x ;...
            left_eye_y,right_eye_y,nip_y,left_mouth_y,right_mouth_y];
        
        image_filename = fullfile(root_folder,C{1,1});
        if exist(image_filename,'file') == 0
            continue;
        end
        img = imread(image_filename);
        if size(img, 3) < 3
           img(:,:,2) = img(:,:,1);
           img(:,:,3) = img(:,:,1);
        end
        [file_folder, file_name, file_ext] = fileparts(image_filename);
        target_filename = strrep(image_filename,root_folder, target_folder);
        assert(strcmp(target_filename, image_filename)==0);
        [file_folder, file_name, file_ext] = fileparts(target_filename);
        if exist(file_folder,'dir') == 0
            mkdir(file_folder);
        end;
        if exist(target_filename) ~= 0
            continue;
        end
        disp([num2str(image_id) '/' num2str(total_num) ' ' target_filename]);
       Tfm =  cp2tform(facial5points', coord5points', 'similarity');
       cropImg = imtransform(img, Tfm, 'XData', [1 imgSize(2)],...
                                      'YData', [1 imgSize(1)], 'Size', imgSize);
       imwrite(cropImg,target_filename);       
    end  
    fclose(csv_fid);



end