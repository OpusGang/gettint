## Get Tint

This script helps identify correct fix for a tint difference between two sources.
It first attempts to apply the following common errors:
 - 2.2 / 2.5 QuickTime "gamma bug"
 - TV to PC levels
 - PC to TV levels
 - 0-235 / 0-240 levels
 - 16-255 levels
 - BT.601 to BT.709 color matrix
 - BT.709 to BT.601 color matrix
 - 10-bit to 8-bit truncation

If fixing these does not result in low enough mean differences, it falls back to attempting to identify the tint with `matchcolors` with the following parameters:
 - gamma
 - offset and gain
 - levels

Keep in mind that the results from `matchcolors` are to be taken with a grain of salt.
The script tends to predict gamma most commonly and struggles the most with level adjustments.
If results are unsatisfactory, it can help to manually run `matchcolors` with different params or just accept that the grade isn't fixable by a script this simple.

When passing YUV video, autocrop is applied with standard ranges.
RGB images must be cropped properly before passing to the script.
Both black bars and texts need to be cropped.

## Match Colors

This is a super basic script to aid with detinting sources that have had simple color corrections or level adjustments performed on them.

For this to work properly, black bars must be cropped, and clip contents must match exactly. A low-pass filter is applied by default to help with noise-related fluctuations.

### Contact
- **IRC Channel**: `#OpusGang` on `irc.libera.chat`
