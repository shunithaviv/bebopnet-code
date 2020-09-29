## How to modify soundfont

Open `/etc/timidity/timidity.cfg`.
( or `/usr/local/share/timidity/` for timidity 2.14 )

Add the following line: `soundfont /path/to/soundfont.sf2`

Notice: midi in xml file has to be set to:
        <midi-program>1</midi-program>