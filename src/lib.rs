use std::{fmt::Display, io::Read, io::Write};

#[derive(Debug)]
pub enum Error {
    BadHeaderLength,
    BadHeaderMagicBytes,
    BadHeaderInvalidChannels,
    BadHeaderInvalidColorSpace,
    UnexpectedEndOfInput,
    InvalidTrailer,
    IoError(std::io::Error),
}

impl Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::BadHeaderLength => f.write_str("Bad header, insufficient length"),
            Error::BadHeaderMagicBytes => f.write_str("Bad header, invalid magic bytes"),
            Error::BadHeaderInvalidChannels => f.write_str("Bad header, invalid channels field"),
            Error::BadHeaderInvalidColorSpace => {
                f.write_str("Bad header, invalid colorspace field")
            }
            Error::UnexpectedEndOfInput => f.write_str("Unexpected end of input"),
            Error::InvalidTrailer => f.write_str("Invalid trailer"),
            Error::IoError(io_error) => {
                f.write_fmt(format_args!("Unexpected IO error {}", io_error))
            }
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        match error.kind() {
            std::io::ErrorKind::UnexpectedEof => Error::UnexpectedEndOfInput,
            _ => Error::IoError(error),
        }
    }
}

#[repr(u8)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Channels {
    RGB = 3,
    RGBA = 4,
}

impl TryFrom<u8> for Channels {
    type Error = Error;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            3 => Ok(Channels::RGB),
            4 => Ok(Channels::RGBA),
            _ => Err(Error::BadHeaderInvalidChannels),
        }
    }
}

#[repr(u8)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ColorSpace {
    SRGBWithLinearAlpha = 0,
    Linear = 1,
}

impl TryFrom<u8> for ColorSpace {
    type Error = Error;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(ColorSpace::SRGBWithLinearAlpha),
            1 => Ok(ColorSpace::Linear),
            _ => Err(Error::BadHeaderInvalidColorSpace),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Header {
    width: u32,
    height: u32,
    channels: Channels,
    colorspace: ColorSpace,
}

const MAGIC: [u8; 4] = [b'q', b'o', b'i', b'f'];

impl Header {
    fn read<T: Read>(reader: &mut T) -> Result<Header, Error> {
        let mut buffer = [0_u8; 14];
        let read_result = reader.read_exact(&mut buffer);
        if read_result.is_ok() {
            if buffer.starts_with(&MAGIC) {
                let width = u32::from_be_bytes(buffer[4..8].try_into().unwrap());
                let height = u32::from_be_bytes(buffer[8..12].try_into().unwrap());
                let channels = buffer[12].try_into()?;
                let colorspace = buffer[13].try_into()?;
                Ok(Header {
                    width,
                    height,
                    channels,
                    colorspace,
                })
            } else {
                Err(Error::BadHeaderMagicBytes)
            }
        } else {
            Err(Error::BadHeaderLength)
        }
    }

    /// length of the buffer required for decoding
    #[inline(always)]
    pub fn decode_buffer_length(&self) -> usize {
        self.width as usize * self.height as usize * self.channels as usize
    }
}

impl From<Header> for [u8; 14] {
    fn from(header: Header) -> Self {
        let mut out = [0; 14];
        (&mut out[0..4]).copy_from_slice(&MAGIC);
        (&mut out[4..8]).copy_from_slice(&header.width.to_be_bytes());
        (&mut out[8..12]).copy_from_slice(&header.height.to_be_bytes());
        out[12] = header.channels as u8;
        out[13] = header.colorspace as u8;
        out
    }
}

#[repr(transparent)]
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq)]
struct Pixel([u8; 4]);

impl Pixel {
    const OPAQUE: [u8; 4] = [0, 0, 0, 255];

    #[inline(always)]
    const fn start() -> Self {
        Self(Pixel::OPAQUE)
    }

    #[inline(always)]
    const fn from_rgba_buffer(buffer: &[u8]) -> Self {
        Self([buffer[0], buffer[1], buffer[2], buffer[3]])
    }

    #[inline(always)]
    const fn from_rgb_buffer_and_alpha(buffer: &[u8], alpha: u8) -> Self {
        Self([buffer[0], buffer[1], buffer[2], alpha])
    }

    #[inline(always)]
    const fn hash(&self) -> usize {
        (self.r() as usize * 3
            + self.g() as usize * 5
            + self.b() as usize * 7
            + self.a() as usize * 11)
            % 64
    }

    #[inline(always)]
    fn write<T: Write>(&self, channels: Channels, mut writer: T) -> std::io::Result<usize> {
        writer.write(&self.0[0..channels as usize])
    }

    #[inline(always)]
    fn read<T: Read>(channels: Channels, mut reader: T) -> std::io::Result<Pixel> {
        let mut buf = Pixel::OPAQUE;
        reader.read_exact(&mut buf[0..channels as usize])?;
        Ok(Pixel(buf))
    }

    #[inline(always)]
    const fn r(&self) -> u8 {
        self.0[0]
    }

    #[inline(always)]
    fn r_mut(&mut self) -> &mut u8 {
        &mut self.0[0]
    }

    #[inline(always)]
    const fn g(&self) -> u8 {
        self.0[1]
    }

    #[inline(always)]
    fn g_mut(&mut self) -> &mut u8 {
        &mut self.0[1]
    }

    #[inline(always)]
    const fn b(&self) -> u8 {
        self.0[2]
    }

    #[inline(always)]
    fn b_mut(&mut self) -> &mut u8 {
        &mut self.0[2]
    }

    #[inline(always)]
    const fn a(&self) -> u8 {
        self.0[3]
    }
}

const OP_RGBA: u8 = 0b11111111;
const OP_RGB: u8 = 0b11111110;
const OP_INDEX: u8 = 0b00000000;
const OP_DIFF: u8 = 0b01000000;
const OP_LUMA: u8 = 0b10000000;
const OP_RUN: u8 = 0b11000000;

#[inline(always)]
const fn op_rgba(pixel: Pixel) -> [u8; 5] {
    [OP_RGBA, pixel.r(), pixel.g(), pixel.b(), pixel.a()]
}

#[inline(always)]
const fn op_rgb(pixel: Pixel) -> [u8; 4] {
    [OP_RGB, pixel.r(), pixel.g(), pixel.b()]
}

#[inline(always)]
const fn op_index(index: usize) -> [u8; 1] {
    [OP_INDEX | (index as u8 & 0b00111111)]
}

#[inline(always)]
const fn op_diff(diffr: i8, diffg: i8, diffb: i8) -> [u8; 1] {
    [OP_DIFF
        | ((diffr + 2) as u8 & 0b00000011) << 4
        | ((diffg + 2) as u8 & 0b00000011) << 2
        | ((diffb + 2) as u8 & 0b00000011)]
}

#[inline(always)]
const fn op_luma(diffg: i8, diffgr: i8, diffgb: i8) -> [u8; 2] {
    [
        OP_LUMA | ((diffg + 32) as u8 & 0b00111111),
        ((diffgr + 8) as u8 & 0b00001111) << 4 | ((diffgb + 8) as u8 & 0b00001111),
    ]
}

#[inline(always)]
const fn op_run(run: u8) -> [u8; 1] {
    [OP_RUN | ((run - 1) & 0b00111111)]
}

const TRAILER: [u8; 8] = {
    let mut out = [op_index(0)[0]; 8];
    out[7] = op_index(1)[0];
    out
};

pub fn encode<R: Read, W: Write>(
    mut reader: R,
    mut writer: W,
    width: u32,
    height: u32,
    channels: Channels,
    colorspace: ColorSpace,
) -> Result<usize, Error> {
    let mut written = 0;

    let header = Header {
        width,
        height,
        channels,
        colorspace,
    };
    let header_bytes: [u8; 14] = header.into();
    written += writer.write(&header_bytes)?;

    let mut previous = Pixel::start();
    let mut index = [Pixel::default(); 64];

    let mut run = 0;
    for _ in 0..width * height {
        let pixel = Pixel::read(channels, &mut reader)?;
        if pixel == previous {
            run += 1;
            if run == 63 {
                written += writer.write(&op_run(62))?;
                run = 1;
            }
        } else {
            if run > 0 {
                written += writer.write(&op_run(run))?;
                run = 0;
            }

            let i = pixel.hash();
            if index[i] == pixel {
                // if the pixel is stored in the index, use that
                written += writer.write(&op_index(i))?;
            } else if pixel.a() == previous.a() {
                // check if diffs from the previous pixel fit diff ops, if not
                // we write the color channels
                let diffr = (pixel.r().wrapping_sub(previous.r())) as i8;
                let diffg = (pixel.g().wrapping_sub(previous.g())) as i8;
                let diffb = (pixel.b().wrapping_sub(previous.b())) as i8;
                let diffgr = diffr.wrapping_sub(diffg);
                let diffgb = diffb.wrapping_sub(diffg);
                if diffr > -3 && diffr < 2 && diffg > -3 && diffg < 2 && diffb > -3 && diffb < 2 {
                    written += writer.write(&op_diff(diffr, diffg, diffb))?;
                } else if diffgr > -9
                    && diffgr < 8
                    && diffg > -33
                    && diffg < 32
                    && diffgb > -9
                    && diffgb < 8
                {
                    written += writer.write(&op_luma(diffg, diffgr, diffgb))?;
                } else {
                    written += writer.write(&op_rgb(pixel))?;
                }
            } else {
                // if alpha differs then we write the whole pixel
                written += writer.write(&op_rgba(pixel))?;
            }
            index[i] = pixel;
            previous = pixel;
        }
    }
    if run > 0 {
        written += writer.write(&op_run(run))?;
    }

    written += writer.write(&TRAILER)?;

    Ok(written)
}

const OP_MASK: u8 = 0b11000000;
const ARG_MASK: u8 = 0b00111111;
const DIFF_R_MASK: u8 = 0b00110000;
const DIFF_G_MASK: u8 = 0b00001100;
const DIFF_B_MASK: u8 = 0b00000011;
const DIFF_GR_MASK: u8 = 0b11110000;
const DIFF_GB_MASK: u8 = 0b00001111;

pub fn decode<R: Read, W: Write>(mut reader: R, mut writer: W) -> Result<Header, Error> {
    let header = Header::read(&mut reader)?;
    let mut previous = Pixel::start();
    let mut index = [Pixel::default(); 64];
    let total = header.decode_buffer_length();
    let mut written = 0;
    let mut buffer = [0_u8; 5];

    while written < total {
        reader.read_exact(&mut buffer[0..1])?;
        let (pixel, run) = match buffer[0] {
            OP_RGBA => {
                reader.read_exact(&mut buffer[1..5])?;
                (Pixel::from_rgba_buffer(&buffer[1..5]), 1)
            }
            OP_RGB => {
                reader.read_exact(&mut buffer[1..4])?;
                (
                    Pixel::from_rgb_buffer_and_alpha(&buffer[1..4], previous.a()),
                    1,
                )
            }
            other => match other & OP_MASK {
                OP_INDEX => {
                    let i = (other & ARG_MASK) as usize;
                    (index[i], 1)
                }
                OP_DIFF => {
                    let mut pixel = previous;
                    let diffr = ((other & DIFF_R_MASK) >> 4).wrapping_sub(2);
                    let diffg = ((other & DIFF_G_MASK) >> 2).wrapping_sub(2);
                    let diffb = (other & DIFF_B_MASK).wrapping_sub(2);
                    *pixel.r_mut() = pixel.r().wrapping_add(diffr);
                    *pixel.g_mut() = pixel.g().wrapping_add(diffg);
                    *pixel.b_mut() = pixel.b().wrapping_add(diffb);
                    (pixel, 1)
                }
                OP_LUMA => {
                    reader.read_exact(&mut buffer[1..2])?;
                    let mut pixel = previous;
                    let diffg = (other & ARG_MASK).wrapping_sub(32);
                    let diffr = ((buffer[1] & DIFF_GR_MASK) >> 4)
                        .wrapping_sub(8)
                        .wrapping_add(diffg);
                    let diffb = (buffer[1] & DIFF_GB_MASK)
                        .wrapping_sub(8)
                        .wrapping_add(diffg);
                    *pixel.r_mut() = pixel.r().wrapping_add(diffr);
                    *pixel.g_mut() = pixel.g().wrapping_add(diffg);
                    *pixel.b_mut() = pixel.b().wrapping_add(diffb);
                    (pixel, 1)
                }
                OP_RUN => (previous, (other & ARG_MASK) + 1),
                _ => unreachable!(),
            },
        };
        previous = pixel;
        index[pixel.hash()] = pixel;
        for _ in 0..run {
            written += pixel.write(header.channels, &mut writer)?;
        }
    }

    let mut trailer_buffer = [0_u8; 8];
    reader.read_exact(&mut trailer_buffer)?;
    if trailer_buffer != TRAILER {
        Err(Error::InvalidTrailer)
    } else {
        Ok(header)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(unused)]
    #[derive(Debug, Default)]
    pub struct OpCounter {
        rgb: u32,
        rgba: u32,
        index: u32,
        diff: u32,
        luma: u32,
        run: u32,
    }

    #[test]
    fn images_test() -> Result<(), Error> {
        let mut path = std::env::current_dir()?;
        path.push("qoi_test_images");
        for entry in std::fs::read_dir(path)? {
            let path = entry?.path();
            if path.is_file() && path.extension().unwrap() == "qoi" {
                // println!("{:?}", &path);

                // first load up our "golden" pdf bytes
                let mut png_path = path.clone();
                assert!(png_path.set_extension("png"));
                let file = std::fs::File::open(png_path)?;
                let png_decoder = png::Decoder::new(file);
                let mut reader = png_decoder.read_info().unwrap();
                let mut buf = vec![0; reader.output_buffer_size()];
                let png_info = reader.next_frame(&mut buf).unwrap();
                let bytes = &buf[..png_info.buffer_size()];

                let channels = match png_info.color_type {
                    png::ColorType::Rgb => Channels::RGB,
                    png::ColorType::Rgba => Channels::RGBA,
                    _ => unreachable!(),
                };

                // now decode the correct qoi file
                let file = std::fs::File::open(&path)?;
                let mut decoded_bytes = Vec::with_capacity(reader.output_buffer_size());
                let loaded_header = decode(&file, &mut decoded_bytes)?;
                assert_eq!(loaded_header.width, png_info.width);
                assert_eq!(loaded_header.height, png_info.height);
                assert_eq!(loaded_header.channels, channels);
                // always this
                assert_eq!(loaded_header.colorspace, ColorSpace::SRGBWithLinearAlpha);
                assert_eq!(bytes, &decoded_bytes);

                // now encode it
                let mut encoded_bytes = Vec::new();
                encode(
                    bytes,
                    &mut encoded_bytes,
                    png_info.width,
                    png_info.height,
                    channels,
                    ColorSpace::SRGBWithLinearAlpha,
                )?;

                // and decode again, should be the same
                let mut decoded_bytes = Vec::with_capacity(reader.output_buffer_size());
                let decoded_header = decode(&encoded_bytes[..], &mut decoded_bytes)?;
                assert_eq!(decoded_header, loaded_header);
                assert_eq!(bytes, &decoded_bytes);
            }
        }

        Ok(())
    }
}
