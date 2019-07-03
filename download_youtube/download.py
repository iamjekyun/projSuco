from __future__ import unicode_literals
import youtube_dl

# for MacOS
# brew install libav
# curl -L https://yt-dl.org/downloads/latest/youtube-dl -o /usr/local/bin/youtube-dl
# chmod a+rx /usr/local/bin/youtube-dl
# brew install ffmpeg
# sudo -H pip install --upgrade youtube-dl

PATH_SAVE = './SaveFile'
FILE_LIST = './fileList.txt'


class MyLogger(object):
    def debug(self, msg):
        pass
    def warning(self, msg):
        pass
    def error(self, msg):
        print(msg)

def my_hook(d):
    if d['status'] == 'finished':
        print('Done downloading, now converting ...')

ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '192',
    }],
    'logger': MyLogger(),
    # 'progress_hooks': [my_hook],
    # 'outtmpl': './files/%(title)s.%(ext)s'
    'outtmpl': PATH_SAVE+ '/%(title)s.%(ext)s'

}

def main():
    youTube = youtube_dl.YoutubeDL(ydl_opts)

    with open(FILE_LIST) as f:
        totalCount = len(open(FILE_LIST).readlines())    
        i = 0  
        for file_name in f:
            i = i + 1
            URL = file_name.rstrip('\n')
            print(str(i)+'/'+str(totalCount), URL)
            try:
                youTube.download([URL])
            except:
                print('error',URL)
                with open('./errorList.txt', "a") as ff:
                    ff.write(URL + '\n')
                pass

if __name__ == '__main__':
    main()
