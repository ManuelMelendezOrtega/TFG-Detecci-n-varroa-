package com.tfg.basico.controller;

import java.time.LocalDate;
import java.util.List;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.security.Principal;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.security.core.Authentication;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.web.servlet.config.annotation.ResourceHandlerRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;
import org.springframework.http.*;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.core.io.ByteArrayResource;

import com.tfg.basico.dto.AnalisisFotoDTO;
import com.tfg.basico.model.*;
import com.tfg.basico.repository.*;

@Controller
public class maincontroller implements WebMvcConfigurer {

    @Autowired private RestTemplate restTemplate;
    @Autowired private SesionMuestreoRepository sesionRepo;
    @Autowired private FotoDetalleRepository fotoRepo;
    @Autowired private UserRepository userRepo;
    @Autowired private PasswordEncoder passwordEncoder;
    @Autowired private RoleRepository roleRepo; 

    private final String PATH_RESULTS = "../api-python/static/results/";
    private final String PATH_UPLOADS = "../api-python/static/uploads/";



    @GetMapping("/")
    public String paginaPrincipal() { return "index"; }

    @GetMapping("/login")
    public String mostrarLogin() { return "login"; }

    @GetMapping("/registro")
    public String mostrarRegistro() { return "registro"; }

    @PostMapping("/registro")
    public String procesarRegistro(@RequestParam String username, @RequestParam String password, Model model) {
        if (userRepo.findByUsername(username) != null) {
            model.addAttribute("error", "Ese usuario ya existe");
            return "registro";
        }
        User user = new User();
        user.setUsername(username);
        user.setPassword(passwordEncoder.encode(password));
        Role rolUsuario;
        if ("admin".equalsIgnoreCase(username)) {
            rolUsuario = roleRepo.findByRoleName("ADMIN");
            if (rolUsuario == null) {
                rolUsuario = roleRepo.findByRoleName("ROLE_ADMIN"); 
            }
        } else {
            rolUsuario = roleRepo.findByRoleName("USER");
            if (rolUsuario == null) {
                rolUsuario = roleRepo.findByRoleName("ROLE_USER"); 
            }
        }
        
        if (rolUsuario != null) { 
            user.setUserRole(rolUsuario); 
        }

        userRepo.save(user);
        return "redirect:/login?registrado=true";
    }

    @GetMapping("/mis-analisis")
    public String verHistorial(Model model, Authentication authentication) {
        User user = userRepo.findByUsername(authentication.getName());
        List<SesionMuestreo> sesiones = sesionRepo.findByUsuarioOrderByFechaDesc(user);
        List<Map<String, Object>> datosGrafica = sesiones.stream().map(s -> {
            Map<String, Object> map = new HashMap<>();
            map.put("fecha", s.getFecha().toString());
            map.put("media", s.getMediaVarroas());
            map.put("numFotos", s.getNumFotos()); 
            return map;
        }).collect(Collectors.toList());
        model.addAttribute("sesiones", sesiones);
        model.addAttribute("datosGrafica", datosGrafica);
        LocalDate hace21Dias = LocalDate.now().minusDays(21);
        List<SesionMuestreo> sesionesPrevias = sesionRepo.findByUsuarioAndFecha(user, hace21Dias);
        
        if (!sesionesPrevias.isEmpty()) {
            double media21Dias = sesionesPrevias.stream()
                    .mapToDouble(SesionMuestreo::getMediaVarroas)
                    .average()
                    .orElse(0.0);
            model.addAttribute("mediaCicloAnterior", media21Dias);
            model.addAttribute("huboAnalisisPrevio", true);
        } else {
            model.addAttribute("huboAnalisisPrevio", false);
        }
        return "mis_analisis";
    }

    @GetMapping("/nuevo-analisis")
    public String formulario() { return "nuevo_analisis"; }

    @PostMapping("/analizar")
    public String procesarAnalisis(@RequestParam("fecha") String fechaStr, @RequestParam("fotos") MultipartFile[] fotos, Principal principal, Model model) {
        try {
            User usuario = userRepo.findByUsername(principal.getName());
            SesionMuestreo sesion = new SesionMuestreo();
            sesion.setUsuario(usuario);
            sesion.setFecha(LocalDate.parse(fechaStr));
            sesion.setNumFotos(fotos.length);
            sesion.setMediaVarroas(0.0f);
            sesion = sesionRepo.save(sesion);

            int totalVarroas = 0;
            String pythonApiUrl = "http://localhost:5000/api/analizar";

            for (MultipartFile archivo : fotos) {
                if (archivo.isEmpty()) continue;
                try {
                    String prefijo = UUID.randomUUID().toString().substring(0, 8);
                    String nombreOriginal = Paths.get(archivo.getOriginalFilename()).getFileName().toString();
                    String nombreUnico = prefijo + "_" + nombreOriginal;

                    byte[] bytes = archivo.getBytes();

                    HttpHeaders headers = new HttpHeaders();
                    headers.setContentType(MediaType.MULTIPART_FORM_DATA);
                    MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
                    
                    body.add("foto", new ByteArrayResource(bytes) {
                        @Override public String getFilename() { return nombreUnico; }
                    });

                    HttpEntity<MultiValueMap<String, Object>> req = new HttpEntity<>(body, headers);
                    ResponseEntity<AnalisisFotoDTO> res = restTemplate.postForEntity(pythonApiUrl, req, AnalisisFotoDTO.class);
                    
                    if (res.getBody() != null) {
                        totalVarroas += res.getBody().getConteo();
                        FotoDetalle det = new FotoDetalle();
                        det.setSesion(sesion);
                        det.setConteoVarroas(res.getBody().getConteo());
                        
                        det.setRutaImagenOriginal(nombreUnico); 
                        det.setRutaImagenMarcada(res.getBody().getRutaMarcada());
                        
                        fotoRepo.save(det);
                    }
                } catch (Exception e) { System.err.println("Error en foto: " + e.getMessage()); }
            }
            if (fotos.length > 0) {
                sesion.setMediaVarroas((float) totalVarroas / fotos.length);
                sesionRepo.save(sesion);
            }
            return "redirect:/mis-analisis";
        } catch (Exception e) {
            model.addAttribute("error", "Error: " + e.getMessage());
            return "nuevo_analisis";
        }
    }

    @Override
    public void addResourceHandlers(ResourceHandlerRegistry registry) {
        registry.addResourceHandler("/detections/**")
                .addResourceLocations("file:" + PATH_RESULTS);
    }


    @PostMapping("/eliminar-sesion")
    public String eliminarSesion(@RequestParam("id") Integer id, org.springframework.web.servlet.mvc.support.RedirectAttributes ra) {
        try {
            SesionMuestreo sesion = sesionRepo.findById(id).orElse(null);
            if (sesion != null) {
                borrarArchivosFisicosDeSesion(sesion);
                sesionRepo.delete(sesion);
                ra.addFlashAttribute("mensaje", "Eliminado correctamente del disco y base de datos");
            }
        } catch (Exception e) { ra.addFlashAttribute("error", "Error al eliminar"); }
        return "redirect:/mis-analisis";
    }

    @GetMapping("/admin/usuarios")
    public String listarUsuariosAdmin(Model model) {
        model.addAttribute("usuarios", userRepo.findAll());
        return "admin_usuarios";
    }

    @PostMapping("/admin/usuarios/eliminar")
    public String eliminarUsuarioAdmin(@RequestParam("id") Integer id, org.springframework.web.servlet.mvc.support.RedirectAttributes ra) {
        try {
            User usuario = userRepo.findById(id).orElse(null);
            if (usuario != null) {
                List<SesionMuestreo> sesiones = sesionRepo.findByUsuarioOrderByFechaDesc(usuario);
                for (SesionMuestreo sesion : sesiones) {
                    borrarArchivosFisicosDeSesion(sesion);
                    sesionRepo.delete(sesion);
                }
                userRepo.deleteById(id);
                ra.addFlashAttribute("mensaje", "Usuario y todos sus archivos eliminados con éxito");
            }
        } catch (Exception e) { ra.addFlashAttribute("error", "Error: " + e.getMessage()); }
        return "redirect:/admin/usuarios";
    }


    @PostMapping("/admin/limpiar-archivos")
    public String limpiarArchivosResiduales(org.springframework.web.servlet.mvc.support.RedirectAttributes ra) {
        try {
            int borradosResults = ejecutarLimpiezaCarpeta(PATH_RESULTS, true);
            int borradosUploads = ejecutarLimpiezaCarpeta(PATH_UPLOADS, false);
            ra.addFlashAttribute("mensaje", "Limpieza: " + borradosResults + " en Results y " + borradosUploads + " en Uploads.");
        } catch (Exception e) {
            ra.addFlashAttribute("error", "Error en la limpieza: " + e.getMessage());
        }
        return "redirect:/admin/usuarios";
    }

    private void borrarArchivosFisicosDeSesion(SesionMuestreo sesion) {
        if (sesion.getFotos() != null) {
            for (FotoDetalle foto : sesion.getFotos()) {
                try {
                    Files.deleteIfExists(Paths.get(PATH_RESULTS + foto.getRutaImagenMarcada()));
                    Files.deleteIfExists(Paths.get(PATH_UPLOADS + foto.getRutaImagenOriginal()));
                } catch (Exception e) { System.err.println("Error borrando: " + e.getMessage()); }
            }
            fotoRepo.deleteAll(sesion.getFotos());
        }
    }

    private int ejecutarLimpiezaCarpeta(String ruta, boolean esCarpetaResults) throws java.io.IOException {
        int contador = 0;
        Path directorio = Paths.get(ruta);
        if (!Files.exists(directorio)) return 0;
        try (Stream<Path> archivos = Files.list(directorio)) {
            List<Path> lista = archivos.collect(Collectors.toList());
            for (Path p : lista) {
                String nombre = p.getFileName().toString();
                if (Files.isDirectory(p) || nombre.startsWith(".")) continue;
                
                boolean existe = esCarpetaResults ? fotoRepo.existsByRutaImagenMarcada(nombre) 
                                                  : fotoRepo.existsByRutaImagenOriginal(nombre);
                if (!existe) {
                    Files.delete(p);
                    contador++;
                }
            }
        }
        return contador;
    }
}