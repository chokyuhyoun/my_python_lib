PRO ffmpeg, FARRAY, FramesPerSecond, filename=filename, VCODEC=VCODEC, $
            no_buffer=no_buffer, err=err
;+
; Name:ffmpeg
;
; Purpose : make a movie file using image files.
; 
; Input Parameters:
;   FARRAY  - list of image files. 
;             png/jpeg/anything which can read using 'read_image' function are possible.
;             The array should contain the file in order.
;   FramePerSecond  - frame per second of the output movie file.
; 
; Output Parameters :
;   filename  - filename 
;             file will be saved in 'current' directory.
;             recommend 'avi/mov/wmv/mp4' formats for the output file.
;
; History:
;   08-mar-2013 - Heesu Yang
;   13-mar-2013 - Heesu Yang. Check errors before generating the movie.
; 	15-May-2013 - Heesu Yang. mov encoder check.
;   26-Dec-2015 - Heesu Yang. VCODEC keyword added. 
;							  For mp4 extension files, the video codec changed from 'libx264' to 'mpeg4'(native compression).
;						    pix_fmt changed from 'rgba' to 'yuv420'
;   09-Nov-2016 - Kyuhyoun Cho. file copy --> generate list file
;   16-Apr-2018 - Kyuhyoun Cho. Solve error from odd number size image
;   30-Jan-2019 - Kyuhyoun Cho. Add end buffer (1s)
;   13-Dec-2019 - Kyuhyoun Cho. Add out_range=full option to conserve the RGB range
cd, CURRENT=CURRENT_PATH
pngfile_path=FILE_DIRNAME(FILE_EXPAND_PATH(FARRAY[0]))
FARRAY=FILE_BASENAME(FARRAY)

if not keyword_set(FramesPerSecond) then FramesPerSecond=20
if not keyword_set(filename) then filename='out.mp4'
if not keyword_set(FARRAY) then return 
if size(FramesPerSecond, /TYPE) ne 2 then begin
  print, "FramesPerSecond should be integer."
  return
endif

m_filename_extension=strmid(filename, 2, 3, /reverse)
IF KEYWORD_SET(VCODEC) THEN BEGIN 
    m_codec='-codec '+VCODEC
    ENDIF ELSE BEGIN
        if strcmp(m_filename_extension, 'mp4')      then m_codec='-vcodec libx264' $
        else if strcmp(m_filename_extension, 'avi') then m_codec='-vcodec libxvid' $
        else if strcmp(m_filename_extension, 'wmv') then m_codec='' $
        else if strcmp(m_filename_extension, 'mov') then m_codec='-codec mpeg4' $
        else m_codec=''
ENDELSE

fps=strtrim(string(FramesPerSecond),1)

ffmpeg_info=ROUTINE_INFO('ffmpeg', /SOURCE)
ffmpeg_path=FILE_DIRNAME(ffmpeg_info.path)
extension=strmid(FARRAY[0], 2, 3, /reverse)


if strmatch(strmid(farray[0], 2, 3, /reverse), 'txt') eq 0 then begin
  cd, ffmpeg_path 
  openw, 1, 'files.txt'
  for i=0, n_elements(farray)-1 do begin
      printf, 1, "file '"+pngfile_path+path_sep()+farray[i]+"'"
  endfor      
  if ~keyword_set(no_buffer) then begin
    for i=0, framesPerSecond do begin
      printf, 1, "file '"+pngfile_path+path_sep()+farray[-1]+"'"
    endfor
  endif
  close, 1
endif else begin
  cd, current_path
  file_copy, farray, ffmpeg_path+path_sep()+'files.txt', /overwrite
endelse

cd, ffmpeg_path
;cmd='ffmpeg -r '+fps+' -f concat -i files.txt '+m_codec+' -crf 18'+  $
;     ' -vf "fps='+fps+', format=yuv420p, '+ $
;     ' scale=trunc(iw/2)*2:trunc(ih/2)*2"' + $
;     ' -y '+filename
cmd='ffmpeg -r '+fps+' -f concat -i files.txt '+m_codec+' -crf 18'+  $
     ' -vf "fps='+fps+', format=yuv420p, '+ $
     ' scale=trunc(iw/2)*2:trunc(ih/2)*2:out_range=full"' + $
     ' -y '+filename

spawn, cmd, res, err, /log_output
file_delete, 'files.txt'
;stop
;print, cmd
cd, CURRENT_PATH
FILE_MOVE, FILE_SEARCH(ffmpeg_path, filename), CURRENT_PATH, /OVERWRITE, /REQUIRE_DIRECTORY, /ALLOW_SAME 
end