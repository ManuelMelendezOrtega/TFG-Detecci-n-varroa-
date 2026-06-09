package com.tfg.basico.controller;

import java.time.LocalDate;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.security.Principal;
import java.nio.file.*;

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

/**
 * Controlador principal de la aplicación.
 * Conecta las pantallas de la página web con la base de datos y la IA de Python.
 */
@Controller
public class maincontroller implements WebMvcConfigurer {

    // Conexiones automáticas a la base de datos y herramientas de la web
    @Autowired private RestTemplate restTemplate;
    @Autowired private SesionMuestreoRepository sesionRepo;
    @Autowired private FotoDetalleRepository fotoRepo;
    @Autowired private UserRepository userRepo;
    @Autowired private PasswordEncoder passwordEncoder;
    @Autowired private RoleRepository roleRepo; 

    // Carpetas donde se guardan las fotos normales y las fotos analizadas por la IA
    private final String PATH_RESULTS = "../api-python/static/results/";
    private final String PATH_UPLOADS = "../api-python/static/uploads/";

    // --- PÁGINAS PÚBLICAS Y REGISTRO ---

    @GetMapping("/")
    public String paginaPrincipal() { return "index"; }

    @GetMapping("/login")
    public String mostrarLogin() { return "login"; }

    @GetMapping("/registro")
    public String mostrarRegistro() { return "registro"; }

    // Procesa el formulario cuando un apicultor se registra en la web
    @PostMapping("/registro")
    public String procesarRegistro(@RequestParam String username, @RequestParam String password, Model model) {
        // Comprueba si el nombre de usuario ya está cogido
        if (userRepo.findByUsername(username) != null) {
            model.addAttribute("error", "Ese usuario ya existe");
            return "registro";
        }
        
        // Crea el nuevo usuario y encripta su contraseña por seguridad
        User user = new User();
        user.setUsername(username);
        user.setPassword(passwordEncoder.encode(password)); 
        
        // Si el usuario se llama "admin", le da permisos de Administrador; si no, de usuario normal
        Role rolUsuario = "admin".equalsIgnoreCase(username) ? 
                          getRoleSafe("ADMIN") : getRoleSafe("USER");
        if (rolUsuario != null) user.setUserRole(rolUsuario); 

        userRepo.save(user); // Guarda el usuario en la base de datos
        return "redirect:/login?registrado=true";
    }

    private Role getRoleSafe(String baseRole) {
        Role r = roleRepo.findByRoleName(baseRole);
        return r != null ? r : roleRepo.findByRoleName("ROLE_" + baseRole);
    }

    // --- PANEL DEL APICULTOR: HISTORIAL Y ANÁLISIS ---

    // Carga la pantalla "Mis Análisis" con el historial del apicultor y los datos de las gráficas
    @GetMapping("/mis-analisis")
    public String verHistorial(Model model, Authentication authentication) {
        // Busca en la base de datos solo los análisis que pertenecen al usuario que ha iniciado sesión
        User user = userRepo.findByUsername(authentication.getName());
        List<SesionMuestreo> sesiones = sesionRepo.findByUsuarioOrderByFechaDesc(user);
        
        // Adapta los datos de los análisis para que la gráfica de la web los pueda dibujar fácilmente
        List<Map<String, Object>> datosGrafica = sesiones.stream().map(s -> {
            Map<String, Object> map = new HashMap<>();
            map.put("fecha", s.getFecha().toString());
            map.put("media", s.getMediaVarroas());
            map.put("numFotos", s.getNumFotos()); 
            return map;
        }).collect(Collectors.toList());
        
        model.addAttribute("sesiones", sesiones);
        model.addAttribute("datosGrafica", datosGrafica);
        
        // Alerta inteligente: Busca si el apicultor hizo un análisis exactamente hace 21 días
        // (que es el ciclo de cría de las abejas) para comparar si la plaga ha subido o bajado
        LocalDate hace21Dias = LocalDate.now().minusDays(21);
        List<SesionMuestreo> sesionesPrevias = sesionRepo.findByUsuarioAndFecha(user, hace21Dias);
        
        if (!sesionesPrevias.isEmpty()) {
            double media21Dias = sesionesPrevias.stream()
                    .mapToDouble(SesionMuestreo::getMediaVarroas).average().orElse(0.0);
            model.addAttribute("mediaCicloAnterior", media21Dias);
            model.addAttribute("huboAnalisisPrevio", true);
        } else {
            model.addAttribute("huboAnalisisPrevio", false);
        }
        return "mis_analisis";
    }

    @GetMapping("/nuevo-analisis")
    public String formulario() { return "nuevo_analisis"; }

    // Recibe las fotos que sube el usuario, se las envía a la IA en Python y guarda el resultado
    @PostMapping("/analizar")
    public String procesarAnalisis(@RequestParam("fecha") String fechaStr, @RequestParam("fotos") MultipartFile[] fotos, Principal principal, Model model) {
        try {
            // Crea una nueva sesión de muestreo en la base de datos para este día
            User usuario = userRepo.findByUsername(principal.getName());
            SesionMuestreo sesion = new SesionMuestreo();
            sesion.setUsuario(usuario);
            sesion.setFecha(LocalDate.parse(fechaStr));
            sesion.setNumFotos(fotos.length);
            sesion.setMediaVarroas(0.0f);
            sesion = sesionRepo.save(sesion); 

            int totalVarroas = 0;
            String pythonApiUrl = "http://localhost:5000/api/analizar"; // Dirección de la IA de Python

            // Va procesando las fotos subidas una a una
            for (MultipartFile archivo : fotos) {
                if (archivo.isEmpty()) continue;
                try {
                    // Genera un código aleatorio único para el nombre de la foto (así evitamos que una foto machaque a otra)
                    String prefijo = UUID.randomUUID().toString().substring(0, 8);
                    String nombreOriginal = Paths.get(archivo.getOriginalFilename()).getFileName().toString();
                    String nombreUnico = prefijo + "_" + nombreOriginal;

                    // Prepara el paquete con la foto para mandarlo por la red
                    HttpHeaders headers = new HttpHeaders();
                    headers.setContentType(MediaType.MULTIPART_FORM_DATA);
                    MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
                    
                    body.add("foto", new ByteArrayResource(archivo.getBytes()) {
                        @Override public String getFilename() { return nombreUnico; }
                    });

                    // Envía la foto a Python mediante una petición web y espera la respuesta
                    HttpEntity<MultiValueMap<String, Object>> req = new HttpEntity<>(body, headers);
                    ResponseEntity<AnalisisFotoDTO> res = restTemplate.postForEntity(pythonApiUrl, req, AnalisisFotoDTO.class);
                    
                    // Si Python responde correctamente, suma las varroas detectadas y guarda la información en la base de datos
                    if (res.getBody() != null) {
                        totalVarroas += res.getBody().getConteo();
                        FotoDetalle det = new FotoDetalle();
                        det.setSesion(sesion);
                        det.setConteoVarroas(res.getBody().getConteo());
                        det.setRutaImagenOriginal(nombreUnico); 
                        det.setRutaImagenMarcada(res.getBody().getRutaMarcada()); // Foto con los recuadros verdes de la IA
                        fotoRepo.save(det);
                    }
                } catch (Exception e) { System.err.println("Error en foto: " + e.getMessage()); }
            }
            
            // Calcula la media total de ácaros de todo el muestreo y actualiza la sesión
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

    // Le dice a la aplicación dónde están las fotos guardadas en el disco duro para poder mostrarlas en la web
    @Override
    public void addResourceHandlers(ResourceHandlerRegistry registry) {
        registry.addResourceHandler("/detections/**")
                .addResourceLocations("file:" + PATH_RESULTS);
    }

    // --- PANEL DE ADMINISTRACIÓN Y MANTENIMIENTO ---

    // Borra un análisis por completo (tanto de la base de datos como las fotos reales del disco duro)
    @PostMapping("/eliminar-sesion")
    public String eliminarSesion(@RequestParam("id") Integer id, org.springframework.web.servlet.mvc.support.RedirectAttributes ra) {
        try {
            SesionMuestreo sesion = sesionRepo.findById(id).orElse(null);
            if (sesion != null) {
                borrarArchivosFisicosDeSesion(sesion); // Borra los archivos .jpg del ordenador
                sesionRepo.delete(sesion);             // Borra los datos de la base de datos
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

    // El administrador puede borrar a un usuario y automáticamente limpia todas sus fotos y análisis
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

    // Botón de limpieza: Escanea las carpetas y borra cualquier foto "suelta" que no esté registrada en la base de datos
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

    // Método auxiliar para borrar los archivos físicos (.jpg) de una sesión
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

    // Método auxiliar que busca archivos huérfanos en el disco y los destruye para ahorrar espacio
    private int ejecutarLimpiezaCarpeta(String ruta, boolean esCarpetaResults) throws java.io.IOException {
        int contador = 0;
        Path directorio = Paths.get(ruta);
        if (!Files.exists(directorio)) return 0;
        try (Stream<Path> archivos = Files.list(directorio)) {
            List<Path> lista = archivos.collect(Collectors.toList());
            for (Path p : lista) {
                String nombre = p.getFileName().toString();
                if (Files.isDirectory(p) || nombre.startsWith(".")) continue;
                
                // Pregunta a la base de datos si conoce esta foto
                boolean existe = esCarpetaResults ? fotoRepo.existsByRutaImagenMarcada(nombre) 
                                                  : fotoRepo.existsByRutaImagenOriginal(nombre);
                // Si la base de datos no sabe qué foto es, la borra del ordenador
                if (!existe) {
                    Files.delete(p);
                    contador++;
                }
            }
        }
        return contador;
    }
}